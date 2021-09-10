"""
Extracting single image inference to see what this sees.

1. Inference occurs in either of the two models: default 'teacher': paper asserts it's better
2. Each model is a Vision Transformer[https://arxiv.org/abs/2010.11929]
    with an added positional embedding mechanism for self attention

"""
from typing import Union, Any
import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import vision_transformer as vits
from x_utils import apply_cmap


# pylint: disable=no-member
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

norm = lambda x: (x-x.min())/(x.max()-x.min())

def show(x: np.ndarray, width: int=10) -> None:
    
    height = width * x.shape[0]/x.shape[1]
    plt.figure(figsize=(width, height))
    plt.imshow(x)
    plt.show()

class InferDino:
    """light wrapper over Dino inference"""
    def __init__(self, patch_size: int=8, **kwargs):
        self.patch_size = patch_size
        self.model = load_model(patch_size=patch_size, **kwargs)

        self.image = None
        self.attn = None
        self.th_attn = None

    def __exit__(self, exc_type, exc_value, traceback):
        """ cleanup cuda objects
        """
        self.image = None
        self.attn = None
        self.th_attn = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def run(self, image: str, **kwargs) -> None:
        """ process single image file"""
        patch_size = self.patch_size if "patch_size" not in kwargs else kwargs["patch_size"]

        if isinstance(image, str):
            resize=kwargs["resize"] if "resize" in kwargs else None
            crop=kwargs["crop"] if "crop" in kwargs else None
            tensor, self.image = load_image(image, resize=resize, maxsize=1000, crop=crop)
            self.attn, self.th_attn, height, width = infer_single(self.model, tensor, patch_size=patch_size, **kwargs)
            self.image = self.image[:height, :width,...]

        else:
            raise NotImplementedError

    def lump(self, mode: str="concat", cmap: str="inferno", save: str=None) -> np.ndarray:
        """ stacks or masks images and output

        Args
            save: path to save image
        """
        tofloat = lambda x: x.astype(np.float32)/255.
        touint8 = lambda x: (x * 255).astype(np.uint8)
        out = None
        _attn = norm(self.attn.mean(axis=0))
        if mode == "concat" or mode == "all":
            attn = touint8(apply_cmap(_attn, cmap=cmap))
            out =  np.concatenate([self.image, attn], axis=0)
        if mode == "mask" or mode == "all":
            masked  = _attn[..., np.newaxis] * tofloat(self.image)
            masked = touint8(masked)
            out = masked if out is None else np.concatenate([out, masked], axis=0)

        if mode == "gray":
            gray = np.repeat(tofloat(self.image).mean(axis=-1, keepdims=True), 3,axis=-1)
            out = _attn[..., np.newaxis] * tofloat(self.image) + (1 - _attn[..., np.newaxis]) * gray

        if save is not None:
            if osp.splitext(save)[-1].lower() not in (".jpg", ".jpeg", ".png"):
                save += ".jpg"
            Image.fromarray(out).save(save)
        return out

    def batch(self, input_folder:str, output_folder: str) -> None:
        """
        """
        assert osp.isdir(input_folder), f"input folder '{input_folder}' not found"
        images = get_images(input_folder)
        os.makedirs(output_folder, exist_ok=True)
        pbar = tqdm(enumerate(images))
        for i, image in pbar:
            name = osp.basename(image)
            pbar.set_description(f"{i} {name}")
            try:
                self.run(image)
                self.lump(save=osp.join(output_folder, "_attn".join(osp.splitext(name))))

            except:
                print(f"image {image}, fails")

    def show(self, what: str="mean_attention", threshold: bool=False, cmap: str="inferno"):
        att = self.attn if not threshold else self.th_attn
        if what[0] == "m":
            show_attn_mean(att, self.image, cmap=cmap)
        else:
            show_attns(att)

def get_images(folder: str) -> list:
    _get_ext = lambda x: os.path.splitext(x)[-1].lower()
    _img_ext = (".png", ".jpg", ".jpeg")
    return sorted([f.path for f in os.scandir(folder) if _get_ext(f.name) in _img_ext])

def load_model(arch: str="vit_small", patch_size: int=8, pretrained_weights: str="", checkpoint_key: str="teacher", **kwargs) -> Any:
    """ simplified model loader from video_generation

    Args    from argparse.args
        arch        str ["vit_small"] in ["vit_tiny", "vit_small", "vit_base"]
        patch_size  int [8] in tested 5, 8, 16 | 5 best, but slower
        pretrained_weights  str path to pretrained model
        checkpoint_key      str ['teacher'] | 'student'
    """
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(DEVICE)

    if osp.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print( f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")

    else:
        print("Please use the `pretrained_weights` argument to indicate checkpoint path.")
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
              # model used for visualizations in [the] paper, and default args
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("No pretrained weights provided, load reference pretrained DINO weights.")
            url = f"https://dl.fbaipublicfiles.com/dino/{url}"
            state_dict = torch.hub.load_state_dict_from_url(url=url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("No reference weights available for this model => Use random weights.")
    return model

def get_augment(resize: Union[int, tuple, list]=None)-> transforms.Compose:
    """ simple augment - if gpu poops out, resize"""
    out = [transforms.ToTensor()]
    if isinstance(resize, (int, tuple, list)):
        out += [transforms.Resize(resize)]
    out += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    return transforms.Compose(out)

def load_image(path: str, resize=None, maxsize=1000, crop=None) -> tuple:
    """
    Args
        path    (str) image path
        resize  (int, tuple [None])
        maxsize (int [1000]), failsafe resize
        crop    (tuple (y_0, y_1, x_0, x_1))
        # crop uses ndarray order, not ffmpeg, not PIL
 
    """
    assert osp.isfile(path)
    img = Image.open(path).convert("RGB")
    if crop is not None and len(crop) == 4:
        # PIL  crop left, upper, right, lower 
        img = img.crop((crop[2], crop[0], crop[3], crop[1]))

    size = np.asarray(img.size)[::-1]
    _max = np.argmax(size)
    if size[_max] > maxsize and resize is None:
        resize = maxsize
    if isinstance(resize, int):
        resize = ((size * resize/size[_max]).astype(int)).tolist()
    if resize is not None:
        img = img.resize(resize[::-1])
    return get_augment(resize=resize)(img), np.asarray(img)

def infer_single(model: nn.Module, img: Union[str, np.ndarray], patch_size: int=8, threshold: float=0.6, **kwargs) -> tuple:
    """ returns attn, th_attn, width, height
            th_attn:    0.6 mass of each head
            attn:       mean of all heads
    """
    interp = lambda x, s, m="nearest": nn.functional.interpolate(x.unsqueeze(0), scale_factor=s,
                                                                 mode=m,
                                                                 recompute_scale_factor=False
                                                                 )[0].cpu().numpy()

    if isinstance(img, str):
        resize=kwargs["resize"] if "resize" in kwargs else None
        crop=kwargs["crop"] if "crop" in kwargs else None
        img = load_image(img, resize=resize, crop=crop)

    w1, h1 = (img.shape[1] - img.shape[1] % patch_size,
            img.shape[2] - img.shape[2] % patch_size,
    )
    _size = np.asarray(img.shape[1:3])
    w, h = (_size - np.mod(_size, patch_size)).tolist()

    img = img[:, :w, :h].unsqueeze(0)

    w_featmap, h_featmap = (np.asarray(img.shape[-2:]) // patch_size).tolist()

    w_featmap1 = img.shape[-2] // patch_size
    h_featmap1 = img.shape[-1] // patch_size

    assert(w == w1)
    assert(h == h1)
    assert(w_featmap == w_featmap1)
    assert(h_featmap == h_featmap1)

    attentions = model.get_last_selfattention(img.to(DEVICE))

    nh = attentions.shape[1]  # number of heads
    print("attentions:", attentions.shape)

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = interp(th_attn, patch_size, "nearest")

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = interp(attentions, patch_size, "nearest")

    return attentions, th_attn, w, h


def show_attns(attn: np.ndarray, figsize: tuple=(10,10)) -> None:
    """
    """
    x = norm(attn.transpose(1,2,0))
    plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.imshow(x[...,:3])
    plt.subplot(212)
    plt.imshow(x[...,3:6])
    plt.tight_layout()
    plt.show()

def show_attn_mean(attn: np.ndarray, image: np.ndarray=None, figsize: tuple=(10,10), cmap: str="inferno") -> None:
    x = attn.mean(axis=0)
    rows = 2
    if image is None:
        rows = 1
        figsize = (figsize[0], figsize[1]/2)

    plt.figure(figsize=figsize)
    plt.subplot(rows, 1, 1)
    plt.imshow(apply_cmap(x, cmap))
    # plt.imshow(x, cmap=cmap)
    if image is not None:
        plt.subplot(rows, 1, 2)
        plt.imshow(image)
    plt.tight_layout()
    plt.show()

def cat_attn(attn, image, cmap="inferno", save=False, show=True, figsize=None, subplot=None):
    
    x = apply_cmap(attn.mean(axis=0), cmap)
    image = (image/255).astype(x.dtype) if image.dtype == np.uint8 else image
    x = np.stack([image, x], axis=0)

    if isinstance(save, str):
        img = Image.fromarray((x * 255).astype(np.uint8))
        if osp.splitext(save)[-1].lower() not in (".jpg", ".jpeg", ",png"):
            save += ".png"
        img.save(save)
    if show:
        if figsize is not None:
            plt.figure(figsize=figsize)
        if subplot is not None:
            if isinstance(subplot, tuple):
                plt.subplot(*subplot)
            else:
                plt.subplot(subplot)
        plt.imshow(x)
    