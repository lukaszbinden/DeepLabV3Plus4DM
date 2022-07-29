
import platform
import glob
import os

import numpy as np
from numpy import random
import imageio

import torch
from torch.utils.data import Subset
from torchvision import transforms
import torchvision.transforms.functional as tf

from deeplabv3plus.datasets.utils import FileListDataset, TransformedDataset

if platform.node() == 'lars-HP-ENVY-Laptop-15-ep0xxx':
    BASE_PATH = "/home/lars/Outliers/data/BUID"
else:
    BASE_PATH = os.path.expandvars("${TMPDIR}/cityscapes_toy/")

NUM_CLASSES = 2
RESOLUTION = 256

NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
COLOR_JITTER = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)


def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(arr.shape + (NUM_CLASSES,), dtype=np.float32)
    h, w = np.ix_(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    res[h, w, arr] = 1.0

    return res


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[:, :, None]

    return torch.from_numpy(arr.transpose((2, 0, 1))).contiguous()


def training_transform(image, labels):
    if len(labels.shape) > 2:
        labels = labels[:, :, 0]
    
    labels = labels / 255
    labels = one_hot_encoding(labels.astype(int))

    image = tf.to_tensor(image)
    labels = tf.to_tensor(labels)

    image = tf.resize(image, RESOLUTION, interpolation=tf.InterpolationMode.NEAREST)
    labels = tf.resize(labels, RESOLUTION, interpolation=tf.InterpolationMode.NEAREST)

    i, j, th, tw = transforms.RandomCrop.get_params(labels, (RESOLUTION, RESOLUTION))
    image = tf.crop(image, i, j, th, tw)
    labels = tf.crop(labels, i, j, th, tw)

    # as in https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/main.py
    image = COLOR_JITTER(image)

    if torch.rand(1) < 0.5:
        image = tf.hflip(image)
        labels = tf.hflip(labels)

    # cf. https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/main.py
    image = NORMALIZER(image)
    return image, labels.argmax(dim=0)


def training_dataset():
    seg_file_list = sorted(glob.glob(os.path.join(BASE_PATH, "targets/*/*.png")))
    file_list = sorted(glob.glob(os.path.join(BASE_PATH, "images/*/*.png")))

    dataset = FileListDataset(list(zip(file_list, seg_file_list)), loader=lambda x: (imageio.imread(x[0]), imageio.imread(x[1])))
    dataset = TransformedDataset(dataset, training_transform)

    dataset, _ = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))], generator=torch.Generator().manual_seed(1))

    return dataset


def validation_transform(image, labels):
    if len(labels.shape) > 2:
        labels = labels[:, :, 0]

    labels = labels / 255
    labels = one_hot_encoding(labels.astype(int))
        
    image = tf.to_tensor(image)
    labels = tf.to_tensor(labels)

    image = tf.resize(image, RESOLUTION, interpolation=tf.InterpolationMode.NEAREST)
    labels = tf.resize(labels, RESOLUTION, interpolation=tf.InterpolationMode.NEAREST)

    image = tf.center_crop(image, (RESOLUTION, RESOLUTION))
    labels = tf.center_crop(labels, (RESOLUTION, RESOLUTION))

    # cf. https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/main.py
    image = NORMALIZER(image)
    
    return image, labels.argmax(dim=0)


def validation_dataset():
    seg_file_list = sorted(glob.glob(os.path.join(BASE_PATH, "targets/*/*.png")))
    file_list = sorted(glob.glob(os.path.join(BASE_PATH, "images/*/*.png")))

    dataset = FileListDataset(list(zip(file_list, seg_file_list)), loader=lambda x: (imageio.imread(x[0]), imageio.imread(x[1])))
    dataset = TransformedDataset(dataset, validation_transform)

    _, dataset = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))], generator=torch.Generator().manual_seed(1))

    return dataset
