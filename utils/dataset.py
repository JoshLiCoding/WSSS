import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

RESIZE_SIZE = 352
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
])

class CustomVOCSegmentationTrain(Dataset):
    def __init__(self, dataset, paths):
        self.dataset = dataset
        self.pseudolabel_logits_arr = np.load(paths['clipseg_pseudolabels'])
        self.sam_contours_x_arr = np.load(paths['sam_contours_x'])
        self.sam_contours_y_arr = np.load(paths['sam_contours_y'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        transformed_image = transform(image)
        pseudolabel_logits = torch.from_numpy(self.pseudolabel_logits_arr[idx])
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
            transformed_image = transform(image)
            target = torch.from_numpy(target)
            return transformed_image, target