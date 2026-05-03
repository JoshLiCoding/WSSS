import os
import sys
import tarfile
import shutil
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url, check_integrity
from PIL import Image
import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# augmented VOC dataset from https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit_old/VOC2012'
    }
}
# Match OpenAI CLIP ViT pretrained input (e.g. ViT-L/14 uses 224×224).
RESIZE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.Normalize(mean=MEAN, std=STD),
])

def cmap(N=256, normalized=False):
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
    cmap = cmap()
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 n_images=-1):

        is_aug=True
        
        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        self.n_images = n_images
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
            split_f = os.path.join(voc_root, 'train_aug.txt')
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

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
        if self.n_images == -1:
            return len(self.images)
        return min(self.n_images, len(self.images))

    def get_image_class_ids(self, index):
        """Return unique foreground class indices (1..20) present in this image. No image load."""
        mask_path = self.masks[index]
        target = np.array(Image.open(mask_path), dtype=np.int64)
        class_ids = np.unique(target)
        class_ids = class_ids[(class_ids != 0) & (class_ids != 255)]
        return class_ids

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

class COCOSegmentation(data.Dataset):
    """
    COCO 2014 dataset for semantic segmentation.
    Compatible with VOCSegmentation interface.
    """
    cmap = cmap()
    
    def __init__(self,
                 root,
                 year='2014',
                 image_set='train',
                 download=False,
                 transform=None,
                 n_images=-1,
                 things_only=False):
        """
        Args:
            root: Root directory of the COCO dataset
            year: Dataset year ('2014' or '2017')
            image_set: 'train' or 'val'
            download: If True, downloads the dataset (not implemented)
            transform: Optional transform to apply to image and mask
            n_images: Number of images to use (-1 for all)
            things_only: If True, only include "thing" categories (stuff categories become background)
        """
        self.root = os.path.expanduser(root)
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.n_images = n_images
        self.things_only = things_only
        
        if year == '2014':
            if image_set == 'train':
                img_dir = 'train2014'
                ann_file = 'annotations/instances_train2014.json'
            else:  # val
                img_dir = 'val2014'
                ann_file = 'annotations/instances_val2014.json'
        else:  # 2017
            if image_set == 'train':
                img_dir = 'train2017'
                ann_file = 'annotations/instances_train2017.json'
            else:  # val
                img_dir = 'val2017'
                ann_file = 'annotations/instances_val2017.json'
        
        self.img_dir = os.path.join(self.root, img_dir)
        ann_file = os.path.join(self.root, ann_file)
        
        if not os.path.exists(self.img_dir):
            raise RuntimeError(f'Dataset not found at {self.img_dir}. Please download COCO dataset.')
        
        if not os.path.exists(ann_file):
            raise RuntimeError(f'Annotation file not found at {ann_file}')
        
        # Load COCO API
        self.coco = COCO(ann_file)
        
        # Get category information
        self.cat_ids = self.coco.getCatIds()
        self.cats = self.coco.loadCats(self.cat_ids)
        
        # COCO has 80 thing categories, but we'll map them to consecutive indices
        # Background is 0, then categories 1-80
        self.num_classes = len(self.cat_ids) + 1  # +1 for background
        
        # Create mapping from COCO category ID to our class index
        self.cat_id_to_class_id = {0: 0}  # background
        for idx, cat in enumerate(self.cats, start=1):
            self.cat_id_to_class_id[cat['id']] = idx
        
        # Get image IDs that have annotations
        self.ids = list(self.coco.imgs.keys())
        
        # Filter to images that have segmentation annotations
        valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if ann_ids:  # Only keep images with annotations
                valid_ids.append(img_id)
        
        self.ids = valid_ids
        
        # Limit number of images if specified
        if self.n_images > 0:
            self.ids = self.ids[:n_images]

    def get_image_class_ids(self, index):
        """Return unique foreground class indices (1..80) present in this image. Uses COCO API only (no image/mask load)."""
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        class_ids = []
        for ann in anns:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            cat_id = ann['category_id']
            class_id = self.cat_id_to_class_id.get(cat_id, 0)
            if class_id != 0:
                class_ids.append(class_id)
        return np.unique(class_ids).astype(np.int64) if class_ids else np.array([], dtype=np.int64)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Load annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create semantic segmentation mask
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for ann in anns:
            # Skip invalid annotations
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            cat_id = ann['category_id']
            class_id = self.cat_id_to_class_id.get(cat_id, 0)
            
            # Convert COCO polygon/rle to binary mask
            rle = self.coco.annToRLE(ann)
            m = maskUtils.decode(rle)
            mask[m > 0] = class_id
        
        # Convert to PIL Image
        target = Image.fromarray(mask, mode='L')
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def decode_target(cls, mask):
        """Decode semantic mask to RGB image"""
        return cls.cmap[mask]


class CustomSegmentationTrain(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        # Apply augmentations
        # RandomResizedCrop
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.2), ratio=(4. / 5., 5. / 4.))
        image = transforms.functional.crop(image, i, j, h, w)
        target = transforms.functional.crop(target, i, j, h, w)

        image = transforms.functional.resize(image, (RESIZE_SIZE, RESIZE_SIZE), interpolation=Image.BILINEAR)
        target = transforms.functional.resize(target, (RESIZE_SIZE, RESIZE_SIZE), interpolation=Image.NEAREST)

        # RandomHorizontalFlip
        if torch.rand(1) < 0.5:
            image = transforms.functional.hflip(image)
            target = transforms.functional.hflip(target)
        
        # ColorJitter only on image
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        image = color_jitter(image)

        # ToTensor
        image_tensor = transforms.ToTensor()(image)
        target_tensor = torch.from_numpy(np.array(target, dtype=np.int64))

        # Normalize only image
        image_tensor = transforms.Normalize(MEAN, STD)(image_tensor)

        return image_tensor, target_tensor

    def denormalize(self, tensor):
        for t, m, s in zip(tensor, MEAN, STD):
            t.mul_(s).add_(m)
        return tensor

class CustomSegmentationVal(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx): 
        image, target = self.dataset[idx]
        transformed_image = val_transform(image)
        target = torch.from_numpy(np.array(target))
        return transformed_image, target


val_transform_tta = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
class CustomSegmentationValTTA(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx): 
        image, target = self.dataset[idx]
        transformed_image = val_transform_tta(image)
        target = torch.from_numpy(np.array(target))
        return transformed_image, target