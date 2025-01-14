"""
Extracting single image inference to see what this sees.

1. Inference occurs in either of the two models: default 'teacher': paper asserts it's better
2. Each model is a Vision Transformer[https://arxiv.org/abs/2010.11929]
    with an added positional embedding mechanism for self attention

>>> d = InferDino(arch="vit_base")
    Peak GPU Used by model 382 MB
    Created model 'vit_base' with patch_size [8], num params 85,807,872
>>> d.run(file_attn.jpg', as size (3, 1000, 833)
    Attentions: shape torch.Size([1, 12, 13001, 13001])
    peak GPU Use  16136 MB

>>> d = InferDino(arch="vit_small")
    Peak GPU Used by model 90 MB
    Created model 'vit_small' with patch_size [8], num params 21,670,272
>>> d.run(file_attn.jpg', as size (3, 1000, 833)
    Attentions: shape torch.Size([1, 6, 13001, 13001])
    peak GPU Use  7990 MB


"""
from typing import Union, Any
import logging
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
from x_utils import apply_cmap, get_images


# pylint: disable=no-member
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

norm = lambda x: (x-x.min())/(x.max()-x.min())

def show(x: Union[str, np.ndarray, torch.Tensor], width: int=10) -> None:
    if isinstance(x, str):
        x = np.asarray(Image.open(x).convert("RGB"))
    elif isinstance(x, torch.Tensor):
        x = x.cpu().clone().detach().numpy()
        if x.min() < 0:
            x = norm(x)
    height = width * x.shape[0]/x.shape[1]
    plt.figure(figsize=(width, height))
    plt.imshow(x)
    plt.show()


def get_logger(level=logging.INFO):
    logging.basicConfig(format="%(message)s", level=level)
def switch_level(level):
    logging.getLogger().setLevel(level)

class InferDino:
    """light wrapper over Dino inference
    Examples:
    >>> import InferDino, show
    >>> self = InferDino()

    # process batch of images
    >>> self.batch(<from_folder>, <to_folder>)

    # process single image
    >>> self.run(<image_name>)
    >>> show(self.lump())

    # process features

    """
    def __init__(self, patch_size: int=8, loglevel=20, arch="vit_small", **kwargs):
        get_logger(loglevel)
                
        self.patch_size = patch_size
        self.model = load_model(patch_size=patch_size, arch=arch, **kwargs)

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
            logging.info(f"Loaded image '{osp.basename(image)}', as size {tuple(tensor.shape)}")
            self.attn, self.th_attn, height, width = infer_single(self.model, tensor, patch_size=patch_size, **kwargs)
            self.image = self.image[:height, :width,...]

        else:
            raise NotImplementedError

    def get_features(self, image:str, **kwargs) -> None:

        patch_size = self.patch_size if "patch_size" not in kwargs else kwargs["patch_size"]
        resize=kwargs["resize"] if "resize" in kwargs else None
        crop=kwargs["crop"] if "crop" in kwargs else None
        tensor, self.image = load_image(image, resize=resize, maxsize=1000, crop=crop)
        if tensor.ndim == 3:
            tensor = tensor.view(1, *tensor.shape)
        tensor = tensor.to(device=DEVICE)

        feats = self.model.get_intermediate_layers(tensor, n=1)[0].clone()
        return feats

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
            axis = np.argmin(self.image.shape[:2])
            out =  np.concatenate([self.image, attn], axis=axis)
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

    def batch(self, inputs: Union[str, list, tuple], output_folder: str) -> None:
        """ simple load without multiprocess
        args
            input           (str folder | list files)
            output_folder   (str)
        """
        if isinstance(inputs, str):
            assert osp.isdir(inputs), f"input folder '{inputs}' not found"
            inputs = get_images(inputs)
        assert isinstance(inputs, (list, tuple)), f"expected image list got {type(inputs)}"

        os.makedirs(output_folder, exist_ok=True)
        pbar = tqdm(enumerate(inputs))
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

    _peak_mem=torch.cuda.memory_stats(device=None)["reserved_bytes.all.peak"] >> 20

    _params = sum([p.numel() for p in model.parameters()])
    logging.info(f"Created model '{arch}' with patch_size [{patch_size}], num params {_params:,}, peak GPU{_peak_mem} MB")

    if osp.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            logging.info( f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")

    else:
        logging.info("Please use the `pretrained_weights` argument to indicate checkpoint path.")
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
            logging.info("No pretrained weights provided, load reference pretrained DINO weights.")
            url = f"https://dl.fbaipublicfiles.com/dino/{url}"
            state_dict = torch.hub.load_state_dict_from_url(url=url)
            model.load_state_dict(state_dict, strict=True)
        else:
            logging.info("No reference weights available for this model => Use random weights.")
    return model

def get_augment(resize: Union[int, tuple, list]=None)-> transforms.Compose:
    """ simple augment - if gpu poops out, resize"""
    out = [transforms.ToTensor()]
    if isinstance(resize, (int, tuple, list)):
        out += [transforms.Resize(resize)]
    out += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    return transforms.Compose(out)

def load_image(path: str, resize: Union[int, tuple, list, np.ndarray]=None, maxsize: int=1000,
               crop: Union[list, tuple, np.ndarray]=None) -> tuple:
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

@torch.no_grad()
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

    if logging.getLogger().level == logging.DEBUG:
        w1, h1 = (img.shape[1] - img.shape[1] % patch_size,
                img.shape[2] - img.shape[2] % patch_size, )

    _size = np.asarray(img.shape[1:3])
    w, h = (_size - np.mod(_size, patch_size)).tolist()

    img = img[:, :w, :h].unsqueeze(0)

    w_featmap, h_featmap = (np.asarray(img.shape[-2:]) // patch_size).tolist()

    if logging.getLogger().level == logging.DEBUG:
        w_featmap1 = img.shape[-2] // patch_size
        h_featmap1 = img.shape[-1] // patch_size
        assert(w == w1)
        assert(h == h1)
        assert(w_featmap == w_featmap1)
        assert(h_featmap == h_featmap1)

    attentions = model.get_last_selfattention(img.to(DEVICE))

    nh = attentions.shape[1]  # number of heads
    logging.info(f"Attentions: shape {attentions.shape}")

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

    _peak_mem = torch.cuda.memory_stats(device=None)["reserved_bytes.all.peak"] >> 20
    logging.info(f"peak GPU Use  {_peak_mem} MB")

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
    
def compare_models(images, archs=['vit_small', "vit_base"], patch_sizes=[8,16], sizes=[1024,512,256,128],
                   outf='/media/z/Malatesta/SelfAttention/arch_tests'):
    try:
        for i, image in enumerate(images):
            name = osp.splitext(osp.basename(image))[0]
            out = osp.join(outf, name)
            os.makedirs(out, exist_ok=True)
            for a, arch in enumerate(archs):
                for p, patch_size in enumerate(patch_sizes):
                    D = InferDino(arch=arch, patch_size=patch_size)
                    for s, size in enumerate(sizes):
                        D.run(image, resize=size)
                        D.lump(save=osp.join(out, f"{name}_{arch}_{patch_size}_{size}.jpg"))
    except Exception as e:
        logging.error("failed on compare models")
    finally:
        if D is not None:
            del D
        torch.cuda.synchronize()
        torch.cuda.empty_cache()