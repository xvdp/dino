"""
Extracting single image inference for analysis of whats happening here

1. Inference occurs in either of the two models: default 'teacher': paper asserts it's better
2. Each model is a Vision Transformer[https://arxiv.org/abs/2010.11929]
    with an added positional embedding mechanism for self attention

"""
import os.path as osp
import numpy as np
from PIL import Image
from numpy.lib.arraysetops import isin
import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import vision_transformer as vits


# pylint: disable=no-member
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class InferDino:
    def __init__(self, patch_size=8, **kwargs):
        self.patch_size = patch_size
        self.model = load_model(patch_size=patch_size, **kwargs)

        self.image = None
        self.attn = None
        self.th_attn = None

    def run(self, image, **kwargs):
        patch_size = self.patch_size if "patch_size" not in kwargs else kwargs["patch_size"]

        if isinstance(image, str):
            resize=kwargs["resize"] if "resize" in kwargs else None
            tensor, self.image = load_image(image, resize)
            self.attn, self.th_attn = infer_single(self.model, tensor, patch_size=patch_size)

        else:
            raise NotImplementedError

    def show(self, what="mean_attenion", threshold=False):
        att = self.attn if not threshold else self.th_attn
        if what[0] == "m":
            show_attn_mean(att, self.image)
        else:
            show_attns(att)


def load_model(arch="vit_small", patch_size=8, pretrained_weights="", checkpoint_key="teacher", **kwargs):
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

def get_augment(resize=None):
    """ simple augment - if gpu poops out, resize"""
    out = [transforms.ToTensor()]
    if isinstance(resize, (int, tuple, list)):
        out += [transforms.Resize(resize)]
    out += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    return transforms.Compose(out)

def load_image(path: str, resize=None):
    assert osp.isfile(path)
    img = Image.open(path).convert("RGB")
    if isinstance(resize, int):
        size = np.asarray(img.size)[::-1]
        _max = np.argmax(size)
        resize = ((size * resize/size[_max]).astype(int)).tolist()
    return get_augment(resize=resize)(img), np.asarray(img)

def infer_single(model: nn.Module, img, patch_size=8, threshold=0.6):
    """
    threshold   0.6 part of the mass kept for visualization
    """
    interp = lambda x, s, m="nearest": nn.functional.interpolate(x.unsqueeze(0), scale_factor=s, mode=m)[0].cpu().numpy()

    if isinstance(img, str):
        img = load_image(img)

    w, h = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img.to(DEVICE))

    nh = attentions.shape[1]  # number of head

    print(attentions.shape)

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

    # arr = attentions.mean(axis=0)

            # # save attentions heatmaps
            # # mean of all attention heads
            
            # fname = os.path.join(out, "attn-" + os.path.basename(img_path))
            # plt.imsave(
            #     fname=fname,
            #     arr=sum(
            #         attentions[i] * 1 / attentions.shape[0]
            #         for i in range(attentions.shape[0])
            #     ),
            #     cmap="inferno",
            #     format="jpg",
            # )

    return attentions, th_attn


norm = lambda x: (x-x.min())/(x.max()-x.min())

def show_attns(attn, figsize=(10,10)):
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

def show_attn_mean(attn, image=None, figsize=(10,10), cmap="inferno"):
    x = attn.mean(axis=0)
    rows = 2
    if image is None:
        rows = 1
        figsize = (figsize[0], figsize[1]/2)

    plt.figure(figsize=figsize)
    plt.subplot(rows, 1, 1)
    plt.imshow(x, cmap=cmap)
    if image is not None:
        plt.subplot(rows, 1, 2)
        plt.imshow(image)
    plt.tight_layout()
    plt.show()
