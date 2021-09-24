"""
repurposing of copy_detection.py to find similar images on local drives

input: image, folders
 -> list of closest matches,


try: similarity

"""
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
from x_utils import get_images


# pylint: disable=no-member

###
#
# common functions
#
def verify_image(name):
    try:
        im = Image.open(name)
        return True
    except:
        return False
def verify_images(images):
    return [im for im in images if verify_image(im)]

def get_logger(level=logging.INFO):
    logging.basicConfig(format="%(message)s", level=level)

def switch_level(level):
    logging.getLogger().setLevel(level)

def load_image(path: str, resize: Union[int, tuple, list, np.ndarray]=None, maxsize: int=1000,
               crop: Union[list, tuple, np.ndarray]=None,
               transform: transforms.Compose=None, other: dict=None) -> tuple:
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
        transform = get_transform(np.asarray(img.size), resize=resize, maxsize=maxsize, other=other)

    return transform(img)

def get_transform(img_size: Union[tuple, list, np.ndarray],
                  resize: Union[int, tuple, list, np.ndarray]=None,
                  maxsize: int=1000, other:dict=None)-> transforms.Compose:
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
    if other is not None:
        for trans in other:
            out += [transforms.__dict__[trans](**other[trans])]
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
        recursive = None if "recursive" not in kwargs else kwargs["recursive"]

        if folders is not None:
            self.get_image_dset(folders, recursive=recursive)
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

    def get_image_dset(self, folders, recursive=False):
        self.image_dset = ImgListDataset(get_images(folders, recursive=recursive), maxsize=self.maxsize)
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
                self.images = [key for key in _features if 'features' in _features[key]]
                self.features = torch.stack([_features[key]['features'] for key in self.images])

                # reserve key 'model_info' for paramters, arch, batch_size, patch_size, resize,
                if "model_info" in _features:
                    self.__dict__.update(_features["model_info"])
                    if "arch" in _features["model_info"]:
                        self.get_model(arch=_features["model_info"]["arch"])
                        
                # print("arch", self.arch)
                # print("batch_size", self.batch_size)
                # print("patch_size", self.patch_size)

            else:
                logging.warning(f"expected picked torch features, found, {type(features_file)}")

    def save_features(self, features_file):
        """ save to hash_folder
        """
        if self.images is not None and self.features is not None and len(self.images) == len(self.features):
            update = {"model_info":{"arch":self.arch, "batch_size":self.batch_size, "patch_size":self.patch_size}}
            if self.resize is not None:
                update["model_info"]["resize"] = self.resize
            hash_folder(self.images, metadata=self.features, metakey="features", save=features_file, update=update)

    def compute_folder_features(self):
        try:
            self.features = []
            with torch.no_grad():
                if self._loader is not None:
                    # this assumes images are ok = TODO assume an image could fail to load
                    for image, index in tqdm(self._loader):
                        _peak_mem=torch.cuda.memory_stats(device=None)["reserved_bytes.all.peak"] >> 20
                        logging.info(f"image[{index}], {tuple(image.shape)}, gPU  {_peak_mem} MB")
                        features = self.model.get_intermediate_layers(image.cuda(), n=1)[0]
                        out = process_features(features, image.shape, self.patch_size).cpu().clone().detach()
                        self.features.append(out)
                self.features = torch.cat(self.features)
        except Exception as e:
            logging.exception("errs, probably out of memory")
        finally:
            self.clean_cuda([features])

    def find_matches(self, image: Union[str, torch.Tensor], topk: int=20, **kwargs):
        self.image = image
        with torch.no_grad():
            if "resize" in kwargs:
                self.resize = kwargs["resize"]

            other = None if "transform" not in kwargs else kwargs["transform"]

            if not torch.is_tensor(self.image):
                self.image = load_image(self.image, self.resize, self.maxsize, other=other).unsqueeze(0)

            features = self.model.get_intermediate_layers(self.image.cuda(), n=1)[0]
            features = process_features(features, self.image.shape, self.patch_size)

            database = nn.functional.normalize(self.features, dim=1, p=2)
            query = nn.functional.normalize(features.cpu().clone().detach(), dim=1, p=2)
            self.clean_cuda([features])

            # similarity
            similarity = torch.mm(query, database.T)
            distances, indices = similarity.topk(topk, largest=True, sorted=True)

            self.matches = distances.tolist()[0], indices.tolist()[0]
            return self.matches

    @staticmethod
    def bestfit(num, height, width):
        """
        num ~= cols * rows
        cols ~= rows * height/width
        num = rows**2 * height/width
        rows = (num * width/height )**1/2
        cols = num /rows
        """
        cols = (num * height/width )**1/2
        rows = num/cols
        # combination of row*col resulting in min wasted area
        rc, rf = int(np.ceil(rows)), int(rows)
        cc, cf = int(np.ceil(cols)), int(cols)
        grid = np.array([[rf,cf],[rc,cf],[rf,cc],[rc,cc]])
        grid = grid[grid.prod(axis=1) >= num] # discard
        rows, cols = grid[np.argmin(grid.prod(axis=1))]

        # hm, something is bungled here, lets just double check, 
        if rows > cols and width > height:
            _cols = rows
            rows = cols
            cols = _cols
    
        return int(rows), int(cols)

    def show_matches(self, width=20, cols=5, dbsize=None, compactdb=True, topk=None, axis=None, **kwargs):
        """
#data
folder="/home/z/data/Self/Multitudes"
results ="/media/z/Malatesta/SelfAttention/matches"

from x_get_similar import find_matches

# not included
from xenv import *
import augment.transforms as at
images = get_images(folder, verbose=False)
pts = get_files(folder, ext=".pt")
at.load_shortcuts()
# get random path, show image, print name
im = rndlist(images);S(O(im));im[0]


# 45deg directional Blur 3%, Resize small side to 800 pix
find_matches(M(B( Rs(O(im), size=800), x=0.03, y=0.0, angle=-0.75))[0], pts[6], compactdb=0, topk=15, save=osp.join(results, "_dBlur".join(osp.splitext(osp.basename(im[0])))) )
# gauss blur 3%
find_matches(M(B( Rs(O(im), size=800), x=0.03, y=0.03))[0], pts[6], compactdb=0, topk=15, save=osp.join(results, "_Blur".join(osp.splitext(osp.basename(im[0])))) )
# Desaturation, rnd minor rotation += pi/10
find_matches(M(R( Sat(Rs(O(im), size=800)), angle=(-0.1,0.1) ))[0], pts[6], compactdb=0, topk=15, save=osp.join(results, "_RSat".join(osp.splitext(osp.basename(im[0])))) )
# Desaturation rnd major rotation > pi/2  < 3pi/2
find_matches(M(R( Sat(Rs(O(im), size=800)), angle=(0.5,1.7) ))[0], pts[6], compactdb=0, topk=15, save=osp.join(results, "_hiRSat".join(osp.splitext(osp.basename(im[0])))) )

find_matches(M(R( Sat(Rs(O(im), size=800)), angle=(-0.5,0.5) ))[0], pts[8], compactdb=0, topk=15, save=osp.join(results, osp.basename(im[0])))
find_matches(M(R( Rs(O(im), size=600), angle=(-0.5,0.5) ))[0], pts[8], compactdb=0, topk=15)

find_matches(img, pts[8], compactdb=0, topk=12)
find_matches(img, pts[8], compactdb=0, topk=12)


        #ims = rndlist(images,20)
        #o = lambda x: np.asarray(Image.open(x).convert("RGB").resize((256,256)))
        oims = lambda X: [o(x) for x in X]


        """
        norm = lambda x: (x-x.min())/(x.max()-x.min())
        num = len(self.matches[0])
        if topk is not None:
            num = min(num, topk)

        # test image
        image = self.image.clone()
        if image.ndim == 4:
            image = image[0]
        image = norm(image.permute(1,2,0).numpy())

        # if dbsize exists, then assume images are square
        # concatenate all results in rows
        if compactdb:

            o = lambda x, size, pad=0, col=255: np.pad(np.asarray(Image.open(x).convert("RGB").resize(size)),
                                                       ((pad, pad), (pad,pad), (0,0)), constant_values=col)

            h, w = image.shape[:2]
            shape = [h, w]
            small = np.argmin(shape)
            shape[small] *= 2.2
            plt.figure(figsize=(width, width*shape[0]/shape[1]))
            plt.subplot(2-small, 1+small, 1)
            plt.imshow(image)

            # find best fit and concatenate
            plt.subplot(2-small, 1+small, 2)
            
            rows, cols = self.bestfit(num, h, w)
            score = np.round(self.matches[0][0], 3)
            images = [o(self.images[self.matches[1][0]], dbsize, 2, 0)]
            images += [o(self.images[self.matches[1][i]], dbsize, 2) for i in range(1, num)]
            images += [255*np.ones_like(images[0]) for i in range(rows*cols - len(images))]
            images = np.concatenate([np.concatenate(images[i*cols:(i+1)*cols], axis=1) for i in range(rows)], axis=0)

            plt.title(f"top match: {score}", loc="left")
            if axis:
                plt.axis("off")
            plt.imshow(images)

        else:
        # just show top 10 matches
            i = 1
            cols = min(cols, num+1)
            rows = int(np.ceil(num/cols))
            figsize=(width, width*rows/cols)

            plt.figure(figsize=figsize)
            plt.subplot(rows, cols, i)
            plt.title("TARGET")
            plt.imshow(image)

            for i in range(num):
                image = self.images[self.matches[1][i]]
                score = np.round(self.matches[0][i], 3)
                plt.subplot(rows, cols, i+2)
                if not i:
                    plt.title(f"top match: {score}")
                else:
                    plt.title(f"{score}")
                img = Image.open(image)
                if dbsize is not None:
                    img = img.resize(dbsize)
                plt.imshow(img)
        plt.tight_layout()
        if "save" in kwargs and isinstance(kwargs["save"], str):
            save = kwargs['save']
            save, ext = osp.splitext(save)
            if ext.lower() not in (".jpg", ".pdf"):
                save += ".jpg"
            plt.savefig(save)

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

#########
# scripts
#
# 1. given a stored pt pass a file find closest matches, show and tell
def find_matches(image, database, fullpath=False, compactdb=True, **kwargs):
    """

    """
    m = Match(arch="vit_small")
    m.load_features(database)
    m.find_matches(image, **kwargs)

    for i in range(len(m.matches[0])):
        name = m.images[m.matches[1][i]]
        if not fullpath:
            name = osp.basename(name)
        print(i, round(m.matches[0][i],2), "\t", m.matches[1][i],  "\t", name)

    # weak - store dbsize inside pickle
    imsz=int(osp.basename(database).split("_")[2])
    m.show_matches(dbsize=(imsz,imsz), compactdb=compactdb, **kwargs)
    return  m.images[m.matches[1][0]]

# 2. make a pt dataset using minsize
def make_dataset(folder, recursive=False, resize=None, maxsize=1000, arch='vit_small', patch_size=8,
                 batch_size=1, savename=None, savefolder=None):
    """ vit_small : 10k images # 39MB
    @ 1024 ~1.5s/it, @ 256 ~7 it/s
    make_dataset(folder, resize=(256,256), batch_size=8, savename="vit_small_256_8.pt")
    make_dataset(folder, resize=(512,512), batch_size=8, savename="vit_small_512_8.pt")
    make_dataset(folder, resize=(1024,1024), batch_size=1, savename="vit_small_1024_8.pt")
    """
    m = None
    try:
        m = Match(arch=arch, folders=folder, resize=resize, patch_size=patch_size, batch_size=batch_size,
                  recursive=recursive)
        m.compute_folder_features()

        if savename is None:
            _size = maxsize
            if resize is not None:
                _size = resize if isinstance(resize, int) else "-".join([str(i) for i in resize])
            savename = "db_{arch}_{patch_size}_{_size}.pt"

        if savefolder is None or osp.abspath(savename) != savename:
            root = folder if isinstance(folder, str) else folder[0]
            savename = osp.join(root, savename)
        
        m.save_features(savename)
    except:
        logging.error("Match failed")
    finally:
        if m is not None:
            del m
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return savename
