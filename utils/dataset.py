import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# augmented VOC dataset from https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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

class VOCSegmentation(data.Dataset):
    cmap = cmap()
    def __init__(self,
                 root,
                 image_set='train'):

        is_aug=True

        self.root = os.path.expanduser(root)
        self.image_set = image_set
        voc_root = os.path.join(self.root, 'VOCdevkit', 'VOC2012')
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError(f'Dataset not found at {voc_root}.')
        
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
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

class COCOSegmentation(data.Dataset):
    """
    COCO 2014 semantic segmentation, using CLIP-ES's split files (train.txt/val.txt,
    lines are "COCO_{split}2014_{id} [label_ids...]") and pre-rendered masks in
    coco_seg_anno/{id}.png (0 = background, 1-80 = foreground classes).
    """
    cmap = cmap()

    def __init__(self,
                 root,
                 image_set='train'):
        self.root = os.path.join(os.path.expanduser(root), 'coco')
        self.image_set = image_set

        img_dir = os.path.join(self.root, 'train2014' if image_set == 'train' else 'val2014')
        mask_dir = os.path.join(self.root, 'coco_seg_anno')
        split_f = os.path.join(self.root, f'{image_set}.txt')

        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Dataset not found at {img_dir}.')
        if not os.path.exists(split_f):
            raise RuntimeError(f'Split file not found at {split_f}')

        with open(split_f, 'r') as f:
            file_names = [line.split()[0] for line in f if line.strip()]

        self.images = [os.path.join(img_dir, name + '.jpg') for name in file_names]
        self.masks = [os.path.join(mask_dir, name.split('_')[-1] + '.png') for name in file_names]
        assert len(self.images) == len(self.masks)

        self.num_classes = 81  # 80 COCO thing categories + background

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


class CustomSegmentationTrain(Dataset):
    def __init__(self, dataset, resize_size):
        self.dataset = dataset
        self.resize_size = resize_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]

        # Apply augmentations
        # RandomResizedCrop
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.2), ratio=(4. / 5., 5. / 4.))
        image = transforms.functional.crop(image, i, j, h, w)
        target = transforms.functional.crop(target, i, j, h, w)

        image = transforms.functional.resize(image, (self.resize_size, self.resize_size), interpolation=Image.BILINEAR)
        target = transforms.functional.resize(target, (self.resize_size, self.resize_size), interpolation=Image.NEAREST)

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
    def __init__(self, dataset, resize_size):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size)),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        transformed_image = self.transform(image)
        target = torch.from_numpy(np.array(target))
        return transformed_image, target


class CustomSegmentationValTTA(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        transformed_image = self.transform(image)
        target = torch.from_numpy(np.array(target))
        return transformed_image, target