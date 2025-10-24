import os
import sys
import tarfile
import collections
import shutil
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url, check_integrity
from PIL import Image

# augmented VOC dataset from https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    }
}
RESIZE_SIZE = 448
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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
        if self.n_images == -1:
            return len(self.images)
        return min(self.n_images, len(self.images))

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

class CustomVOCSegmentationTrain(Dataset):
    def __init__(self, dataset, num_classes, sam_cache_path, pseudolabels_path, start_index=0):
        self.dataset = dataset
        self.num_classes = num_classes
        self.sam_cache_path = sam_cache_path
        self.pseudolabels_path = pseudolabels_path
        self.start_index = start_index
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx += self.start_index
        image, target = self.dataset[idx]
        sam_contours_x = np.load(os.path.join(self.sam_cache_path, f'sam_contours_x_{idx}.npy'))
        sam_contours_y = np.load(os.path.join(self.sam_cache_path, f'sam_contours_y_{idx}.npy'))
        pseudolabels = np.load(os.path.join(self.pseudolabels_path, f'pseudolabels_{idx}.npy'))
        class_indices = np.load(os.path.join(self.pseudolabels_path, f'class_indices_{idx}.npy'))

        # Convert pseudolabels to torch tensor and permute to (C, H, W)
        pseudolabels_tensor = torch.from_numpy(pseudolabels).float().permute(2, 0, 1)
        pseudolabels_tensor = F.interpolate(pseudolabels_tensor.unsqueeze(0), size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False)[0]

        # RandomResizedCrop
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.5, 1.5), ratio=(3. / 4., 4. / 3.))
        image = transforms.functional.crop(image, i, j, h, w)
        sam_contours_x = transforms.functional.crop(Image.fromarray(sam_contours_x), i, j, h, w-1)
        sam_contours_y = transforms.functional.crop(Image.fromarray(sam_contours_y), i, j, h-1, w)
        pseudolabels_tensor = transforms.functional.crop(pseudolabels_tensor, i, j, h, w)

        image = transforms.functional.resize(image, (RESIZE_SIZE, RESIZE_SIZE), interpolation=Image.BILINEAR)
        sam_contours_x = transforms.functional.resize(sam_contours_x, (RESIZE_SIZE, RESIZE_SIZE - 1), interpolation=Image.NEAREST)
        sam_contours_y = transforms.functional.resize(sam_contours_y, (RESIZE_SIZE - 1, RESIZE_SIZE), interpolation=Image.NEAREST)
        pseudolabels_tensor = transforms.functional.resize(pseudolabels_tensor, (RESIZE_SIZE, RESIZE_SIZE), interpolation=Image.BILINEAR)

        # RandomHorizontalFlip
        if torch.rand(1) < 0.5:
            image = transforms.functional.hflip(image)
            sam_contours_x = transforms.functional.hflip(sam_contours_x)
            sam_contours_y = transforms.functional.hflip(sam_contours_y)
            pseudolabels_tensor = transforms.functional.hflip(pseudolabels_tensor)
        
        # ColorJitter only on image
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        image = color_jitter(image)

        # ToTensor
        image_tensor = transforms.ToTensor()(image)
        sam_contours_x_tensor = transforms.ToTensor()(np.array(sam_contours_x)).squeeze().float()
        sam_contours_y_tensor = transforms.ToTensor()(np.array(sam_contours_y)).squeeze().float()

        # Normalize only image
        image_tensor = transforms.Normalize(MEAN, STD)(image_tensor)

        # Expand pseudolabels
        full_logits = torch.full((self.num_classes, pseudolabels_tensor.shape[1], pseudolabels_tensor.shape[2]), -1e6, dtype=torch.float32)
        for i, class_idx in enumerate(class_indices):
            full_logits[class_idx] = pseudolabels_tensor[i]
        full_logits[0] = pseudolabels_tensor[len(class_indices)] # background class

        return image_tensor, full_logits, sam_contours_x_tensor, sam_contours_y_tensor

    def denormalize(self, tensor):
        for t, m, s in zip(tensor, MEAN, STD):
            t.mul_(s).add_(m)
        return tensor

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