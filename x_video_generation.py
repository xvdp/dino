"""
xvdp
modification of video_generation to read and write video without going thru images

Changes: 
0. Simplify video loading # ffmpeg (must be installed in system / only tested in Ubuntu 18)
loads video saves to video

1. reduce verbosity
# attentions.mean(axis=0) # in original code
    sum(attentions[i] * 1 / attentions.shape[0] for i in range(attentions.shape[0])

# apply_cmap() same as
    plt.save(<>,cmap=cmap) without matplotlib

Added args
    crop        [None]      # crops input images (y0,y1,x0,x1)
    as_video    [1]         # save directly to vido | 0: save images
    cmap        ["inferno"] # defined in x_utils ( | plasma | viridis | gray)
    frames      [None]      # ([from_frame], to_frame)

"""
import os
import os.path as osp
import sys
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn

from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import psutil

import vision_transformer as vits

from x_utils import apply_cmap
from x_ffwrap import FF, FFcap

# pylint: disable=no-member
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
norm = lambda x: (x-x.min())/(x.max()-x.min())

class VideoGenerator:
    def __init__(self, args):
        self.args = args
        # self.model = None
        # Don't need to load model if you only want a video
        if not self.args.video_only:
            self.model = self.__load_model()


    def run(self):
        if self.args.input_path is None or not os.path.exists(self.args.input_path):
            print(f"Provided input path {self.args.input_path} is non valid.")
            sys.exit(1)

        if os.path.isfile(self.args.input_path):
            self._infer_from_video(self.args.input_path, self.args.output_path,
                                   as_video=self.args.as_video, cmap=self.args.cmap,
                                   crop=self.args.crop, frames=self.args.frames)

    def _get_images(self, inp: str):
        _get_ext = lambda x: os.path.splitext(x)[-1].lower()
        _img_ext = (".png", ".jpg", ".jpeg")
        return sorted([f.path for f in os.scandir(inp) if _get_ext(f.name) in _img_ext])

    @staticmethod
    def _get_augment(resize=None):
        """ simple augment - if gpu poops out, resize
        same as found in original code, less verbose
        """
        out = [pth_transforms.ToTensor()]
        if isinstance(resize, (int, tuple, list)):
            out += [pth_transforms.Resize(resize)]
        out += [pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        return pth_transforms.Compose(out)


    def _infer_from_video(self, inp: str, out:str, as_video=True, cmap="inferno", crop=None, frames=None):
        """
        Args
            crop    (list, tuple) y_start, y_end, x_start, x_end
                    x_end and y_end can be None or int
        """
        in_folder = osp.dirname(osp.abspath(inp))
        out_folder = osp.dirname(out)
        if not osp.isdir(out_folder):
            out = osp.join(in_folder, out)
        print(f"Generating attention video to {out}")
        transform = self._get_augment(self.args.resize)


        v_in = FF(inp)
        width = v_in.stats["width"]
        height = v_in.stats["height"]
        if frames is None:
            frames = [0, v_in.stats['nb_frames']]
        if isinstance(frames, int):
            frames = [0, frames]
        frames = list(frames)
        frames[-1] = min(frames[-1], v_in.stats['nb_frames'])

        if crop is not None:
            assert len(crop) == 4, "list or tuple required with (y_start, y_end, x_start, x_end)"
            crop = list(crop)
            crop[1] = height if crop[1] is None else min(crop[1], height)
            crop[3] = width if crop[3] is None else min(crop[3], width)

        if as_video:
            # process one image to check output size
            attn = self._proc_frame(v_in.to_numpy(0,1)[0], transform, crop, cmap)
            height, width, _ = attn.shape

            with FFcap(name=out, height=height, width=width, rate=v_in.stats["rate"]) as v_out:
                pbar = tqdm(range(frames[0], frames[1]))
                for i in pbar:
                    cpu = psutil.virtual_memory().available >> 20
                    pbar.set_description(f"{i} CPU free {cpu} MB")
                    if cpu < 50:
                        print("out of memory")
                        sys.exit(1)
                    if i: # image 0 already processed
                        img = v_in.to_numpy(i, 1)[0]
                        attn = self._proc_frame(img, transform, crop, cmap)

                    v_out.add_frame(attn)

        else: # output image sequence
            folder = osp.splitext(out)[0]+"_attention"
            os.makedirs(folder, exist_ok=True)
            pbar = tqdm(range(frames[0], frames[1]))
            for i in pbar:
                cpu = psutil.virtual_memory().available >> 20
                pbar.set_description(f"{i} CPU free {cpu} MB")
                if cpu < 50:
                    print("out of memory")
                    sys.exit(1)

                img = v_in.to_numpy(i, 1)[0]
                attn = self._proc_frame(img, transform, crop, cmap)

                Image.fromarray(attn).save(osp.join(folder, "attn-%04d.jpg"%i))

        del v_in


    def _proc_frame(self, img, transform, crop=None, cmap="inferno"):
        """  returns mean attention ready to save
        """
        if crop is not None:
            img = img[crop[0]:crop[1],crop[2]:crop[3]]
        return (apply_cmap(self._infer_img(transform(img)).mean(axis=0),
                           cmap=cmap)*255).astype(np.uint8)

    def _infer_img(self, img: torch.Tensor, thresholded=False):
        w, h = (
            img.shape[1] - img.shape[1] % self.args.patch_size,
            img.shape[2] - img.shape[2] % self.args.patch_size,
        )
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // self.args.patch_size
        h_featmap = img.shape[-1] // self.args.patch_size

        attentions = self.model.get_last_selfattention(img.to(DEVICE))

        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        if thresholded:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - self.args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            attentions = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0),
                    scale_factor=self.args.patch_size,
                    mode="nearest",
                    recompute_scale_factor=False,
                )[0]
                .cpu()
                .numpy()
            )
        else:
            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = (
                nn.functional.interpolate(
                    attentions.unsqueeze(0),
                    scale_factor=self.args.patch_size,
                    mode="nearest",
                    recompute_scale_factor=False
                )[0]
                .cpu()
                .numpy()
            )
        return attentions


    def __load_model(self):
        # build model
        model = vits.__dict__[self.args.arch](
            patch_size=self.args.patch_size, num_classes=0
        )
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(DEVICE)

        if os.path.isfile(self.args.pretrained_weights):
            state_dict = torch.load(self.args.pretrained_weights, map_location="cpu")
            if (
                self.args.checkpoint_key is not None
                and self.args.checkpoint_key in state_dict
            ):
                print(
                    f"Take key {self.args.checkpoint_key} in provided checkpoint dict"
                )
                state_dict = state_dict[self.args.checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    self.args.pretrained_weights, msg
                )
            )
        else:
            print(
                "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
            )
            url = None
            if self.args.arch == "vit_small" and self.args.patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif self.args.arch == "vit_small" and self.args.patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif self.args.arch == "vit_base" and self.args.patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif self.args.arch == "vit_base" and self.args.patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print(
                    "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/dino/" + url
                )
                model.load_state_dict(state_dict, strict=True)
            else:
                print(
                    "There is no reference weights available for this model => We use random weights."
                )
        return model

def parse_args():
    parser = argparse.ArgumentParser("Generation self-attention video")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )
    parser.add_argument(
        "--patch_size", default=8, type=int, help="Patch resolution of the self.model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="""Path to a video file if you want to extract frames
            or to a folder of images already extracted by yourself.
            or to a folder of attention images.""",
    )
    parser.add_argument(
        "--output_path",
        default="./",
        type=str,
        help="""Path to store a folder of frames and / or a folder of attention images.
            and / or a final video. Default to current directory.""",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx percent of the mass.""",
    )
    parser.add_argument(
        "--resize",
        default=None,
        type=int,
        nargs="+",
        help="""Apply a resize transformation to input image(s). Use if OOM error.
        Usage (single or W H): --resize 512, --resize 720 1280""",
    )
    parser.add_argument(
        "--video_only",
        action="store_true",
        help="""Use this flag if you only want to generate a video and not all attention images.
            If used, --input_path must be set to the folder of attention images. Ex: ./attention/""",
    )
    parser.add_argument(
        "--fps",
        default=30.0,
        type=float,
        help="FPS of input / output video. Automatically set if you extract frames from a video.",
    )
    parser.add_argument(
        "--video_format",
        default="mp4",
        type=str,
        choices=["mp4", "avi"],
        help="Format of generated video (mp4 or avi).",
    )
    ##
    # @xvdp extra args
    #
    parser.add_argument(
        "--crop",
        type=int,
        nargs="+",
        help="Optional Crop input --crop y0 y1 x0 x1",
    )
    parser.add_argument(
        "--as_video",
        type=int,
        default=1,
        help="as_video 'Default True', if 0 output frames",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        choices=["inferno", "viridis", "plasma", "gray"],
        help="cmap choice"
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="from_frame to_frame, if single to_frame"
    )
    return parser.parse_args()

if __name__ == "__main__":
    """
    Examples: crop video, save result to images
    $ python x_video_generation.py --input_path <> --output_path <> --crop 10 800 10 1270 --as_video 0

    Examples: retrict frames save result to video
    $ python x_video_generation.py --input_path <> --output_path <> --frames 10 100 --as_video 1
    """
    args = parse_args()
    vg = VideoGenerator(args)
    vg.run()
