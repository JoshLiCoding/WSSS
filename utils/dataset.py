import os
import sys
import tarfile
import collections
import shutil
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url, check_integrity
from PIL import Image

# Augmented VOC dataset from https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    }
}
RESIZE_SIZE = 352
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(size=(RESIZE_SIZE, RESIZE_SIZE), scale=(0.5, 2.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=MEAN, std=STD),
# ])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.Normalize(mean=MEAN, std=STD),
])

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

class VOCSegmentation(data.Dataset):
    cmap = voc_cmap()
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None):

        is_aug=True
        
        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        
        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if is_aug and image_set=='train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join(voc_root, 'train_aug.txt')#'./datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

class CustomVOCSegmentationTrain(Dataset):
    def __init__(self, dataset, paths):
        self.dataset = dataset
        self.pseudolabel_logits_dir = paths['clipseg_cache']
        self.sam_contours_x_arr = np.load(paths['sam_contours_x'])
        self.sam_contours_y_arr = np.load(paths['sam_contours_y'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        transformed_image = train_transform(image)
        pseudolabel_logits = torch.from_numpy(np.load(os.path.join(self.pseudolabel_logits_dir, f"pseudolabel_{idx}.npy")))
        sam_contours_x = transforms.ToTensor()(self.sam_contours_x_arr[idx]).squeeze()
        sam_contours_y = transforms.ToTensor()(self.sam_contours_y_arr[idx]).squeeze()

        return transformed_image, pseudolabel_logits, sam_contours_x, sam_contours_y

class CustomVOCSegmentationVal(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        transformed_image = val_transform(image)
        target = torch.from_numpy(np.array(target))
        return transformed_image, target