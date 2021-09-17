"""
repurposing of copy_detection.py to find similar images on local drives

input: image, folders
 -> list of closest matches,
"""
from sys import maxsize
from typing import Any, Union
import multiprocessing as mp
import logging
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import utils
import vision_transformer as vits

from x_hash import hash_folder


# pylint: disable=no-member

###
#
# common functions
#
def get_images(folders: Union[str, list, tuple], recursive: bool=False) -> list:
    """ return images in folder or folder list
    """
    _images = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

    folders = [folders] if isinstance(folders, str) else folders
    out = []
    for folder in folders:
        out += [f.path for f in os.scandir(folder) if osp.splitext(f.name)[-1].lower() in _images]
    # TODO addrecursive
    return sorted(out)

def get_logger(level=logging.INFO):
    logging.basicConfig(format="%(message)s", level=level)

def switch_level(level):
    logging.getLogger().setLevel(level)

def load_image(path: str, resize: Union[int, tuple, list, np.ndarray]=None, maxsize: int=1000,
               crop: Union[list, tuple, np.ndarray]=None, transform: transforms.Compose=None) -> tuple:
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
    if transform is None:
        transform = get_transform(np.asarray(img.size), resize=resize, maxsize=maxsize)

    return transform(img)

def get_transform(img_size: Union[tuple, list, np.ndarray],
                  resize: Union[int, tuple, list, np.ndarray]=None,
                  maxsize: int=1000)-> transforms.Compose:
    """ simple augment, resize only if explicit or image is too large for GPU
    img_size    in PIL coordinates, x,y
    resize      int, tuple
    maxsize     int max size of larger side
    """
    img_size = np.asarray(img_size)[::-1]
    _max = np.argmax(img_size)
    if img_size[_max] > maxsize and resize is None:
        resize = maxsize
    if isinstance(resize, int):
        resize = ((img_size * resize/img_size[_max]).astype(int)).tolist()

    out = [transforms.ToTensor()]
    if resize is not None:
        out += [transforms.Resize(resize)]
    out += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    return transforms.Compose(out)


class ImgListDataset(torch.utils.data.Dataset):
    """ single image at a time
    to load batch, images have to match size, or resize has to be passed
    mod.
    """
    def __init__(self, images, resize=None, maxsize=1000, transform=None):
        self.transform = transform
        self.images = images
        self.resize = resize
        self.maxsize = maxsize

    def __getitem__(self, i):
        img = load_image(self.images[i], self.resize, self.maxsize, self.transform)
        return img, i

    def __len__(self):
        return len(self.images)

@torch.no_grad()
def process_features(features, sample_size, patch_size):
    """ single gpu/cpu feature processor from single batch
    
    returns tensor sized (batch size, 2 * width*height/patch**2)
    Args
        features        tensor output from  model.get_intermediate_layers(samples, n=1)[0]
        sample_size     tuple, torch.size   size of input batch
        patch_size      int     side of attention patch

    """
    #  first
    b, _, h, w = sample_size
    h = int(h/patch_size)
    w = int(w/patch_size)   # h, w of feature image /
    d = features.shape[-1]  # feature depth, channels at level

    class_prediction = features[:, 0, :] # class prediction per patch x patch feature

    features = features[:, 1:, :].reshape(b, h, w, d).clamp(min=1e-6).permute(0, 3, 1, 2)
    features = nn.functional.avg_pool2d(features.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)

    # concatenate class predictions with clamped, sharpened, mean features
    return torch.cat((class_prediction, features), dim=1)

class Match:
    """

    Example:
    # compute folder, features
    >>> from x_get_similar import Match
    >>> folders = <folder or list of folders> #"/media/z/Malatesta/SelfAttention/Multitudes"
    >>> m = Match(folders=folders, arch="vit_small") # arch default "vit_base"
    >>> m.compute_folder_features()

    # store file list, hashed date_size, content, and m.features
    >>> from x_hash import hash_folder
    >>> feature_dict = hash_folder(m.image_dset.images, metadata=m.features, metakey="features", save="vit_small8_features")
    
    """
    def __init__(self, image: str = None, folders: Union[str, list, tuple] = None,
                 features_file: str=None, patch_size: int=8, maxsize: int=1000,
                 arch: str="vit_base", batch_size: int=1, loglevel: int=20, **kwargs):

        get_logger(loglevel)

        # if process ingle image
        self.image = image

        # if process dset
        self.image_dset = None
        self.images = None # == image_dset.images

        # processed data
        self.features = None    # tensor
        self.matches = None     # tuple, (indices, distances)

        # processors
        self._loader = None
        self.model = None

        # processing parameters
        self.arch = arch
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.maxsize = maxsize
        self.resize = None if "resize" not in kwargs else kwargs["resize"]
        self.transform = None if "transform" not in kwargs else kwargs["transform"]

        if folders is not None:
            self.get_image_dset(folders)
            self.get_loader(batch_size=batch_size)
        
        if features_file is not None:
            self.load_features(features_file)

        if self.model is None:
            self.get_model(arch=arch)


    def get_model(self, arch="vit_base", pretrained_weights="", checkpoint_key="teacher"):
        """
            architectures
            training, checkpoints, etc

            TODO pretrained weights, and checkpoint_key
        
        """
        with torch.no_grad():
            if "vit" in arch:
                self.model  = vits.__dict__[arch](patch_size=self.patch_size, num_classes=0)

            if torch.cuda.is_available():
                self.model.cuda()
            self.model.eval()
            _peak_mem = torch.cuda.memory_stats(device=None)["reserved_bytes.all.peak"] >> 20
            logging.info(f"Peak GPU Used by model {_peak_mem} MB")

        utils.load_pretrained_weights(self.model, pretrained_weights, checkpoint_key, arch, self.patch_size)

    def get_image_dset(self, folders):
        self.image_dset = ImgListDataset(get_images(folders), maxsize=self.maxsize)
        self.image_dset.resize = self.resize
        self.image_dset.transform = self.transform
        # copy to current
        self.images = self.image_dset.images

    def get_loader(self, batch_size=1, num_workers=None):
        """
        if single image saturates GPU, use single process to avoid double loading.
        """
        if num_workers is None:
            num_workers = 1 if batch_size == 1 else mp.cpu_count()//3
        self._loader = torch.utils.data.DataLoader(self.image_dset, batch_size=batch_size,
                                                  num_workers=num_workers, drop_last=False,
                                                  shuffle=False)

    def load_features(self, features_file):
        """ load from x_hash folder
        """
        if features_file is not None:
            if osp.isfile(features_file):
                _features = torch.load(features_file)
                self.features = torch.stack([_features[key]['features'] for key in _features])
                self.images = [name for name in _features if name not in ("features")]

                # reserve key 'model_info' for paramters, arch, batch_size, patch_size, resize,
                if "model_info" in _features:
                    if "arch" in _features["model_info"]:
                        self.get_model(arch=_features["model_info"]["arch"])
                        self.__dict__.update(_features["model_info"])

            else:
                logging.warning(f"expected picked torch features, found, {type(features_file)}")

    def save_features(self, features_file):
        """ save to hash_folder
        """
        if self.images is not None and self.features is not None and len(self.images) == len(self.features):
            update = {"model_info":{"arch":self.arch, "batch_size":self.batch_size, "patch_size":self.patch_size}}
            hash_folder(self.images, metadata=self.features, metakey="features", save=features_file, update=update)


    def compute_folder_features(self):
        with torch.no_grad():
            if self._loader is not None:
                for image, index in tqdm(self._loader):
                    _peak_mem=torch.cuda.memory_stats(device=None)["reserved_bytes.all.peak"] >> 20
                    logging.info(f"image[{index}], {tuple(image.shape)}, gPU  {_peak_mem} MB")
                    features = self.model.get_intermediate_layers(image.cuda(), n=1)[0]
                    self.features.append(process_features(features, image.shape, self.patch_size).cpu().clone().detach())
            self.features = torch.cat(self.features)
        self.clean_cuda([features])

    def find_matches(self, fname: str, topk: int=20):
        self.image = fname
        with torch.no_grad():
            image = load_image(fname, self.resize, self.maxsize, self.transform).unsqueeze(0)
            features = self.model.get_intermediate_layers(image.cuda(), n=1)[0]
            features = process_features(features, image.shape, self.patch_size)

            database = nn.functional.normalize(self.features, dim=1, p=2)
            query = nn.functional.normalize(features.cpu().clone().detach(), dim=1, p=2)
            self.clean_cuda([features])

            # similarity
            similarity = torch.mm(query, database.T)
            distances, indices = similarity.topk(topk, largest=True, sorted=True)

            self.matches = distances.tolist()[0], indices.tolist()[0]
            return self.matches

    def show_matches(self, figsize=(20,10), rows=2, cols=5):

        i = 1
        plt.figure(figsize=figsize)

        plt.subplot(rows, cols, i)
        plt.title(f"target image '{self.image}'")
        plt.imshow(Image.open(self.image))
        for i in range(2, 11):
            image = self.images[self.matches[1][i-2]]
            score = np.round(self.matches[0][i-2], 3)
            plt.subplot(rows, cols, i)
            plt.title(f"{score}")
            plt.imshow(Image.open(image))
        plt.tight_layout()
        plt.show()


    def clean_cuda(self, tensors, full=False):
        """ delete tensors, optional model
            TODO delete peak memory, compute current use, available
        """
        for tensor in tensors:
            del tensor
        if full:
            del self.model

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def find_and_show_matches(image, database="/media/z/Malatesta/SelfAttention/Multitudes/vit_small8_features.pt", fullpath=False):
 
    m = Match(arch="vit_small")
    m.load_features(database)
    m.find_matches(image)
    
    for i in range(len(m.matches[0])):
        name = m.images[m.matches[1][i]]
        if not fullpath:
            name = osp.basename(name)
        print(i, round(m.matches[0][i],2), "\t", m.matches[1][i],  "\t", name)

    m.show_matches()
    return  m.images[m.matches[1][0]]
