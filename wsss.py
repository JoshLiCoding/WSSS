import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from model.deeplab import deeplabv3plus_resnet101
from losses import CollisionCrossEntropyLoss, BLPottsLoss
from vis import vis_sample_img, vis_training_loss

VOC_CLASSES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}
VOC_CLASSES_FLIPPED = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15, "potted plant": 16, "sheep": 17, "sofa": 18, "train": 19, "tv/monitor": 20, "ignore": 255}

NUM_TRAIN_IMAGES = 100
NUM_CLASSES = 21
BATCH_SIZE = 32
NUM_EPOCHS = 300
LEARNING_RATE = 0.01
MOMENTUM = 0.9
IGNORE_INDEX = 255
RESIZE_SIZE = 352
DISTANCE_TRANSFORM = 'euclidean'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIRS = {
    'output': 'output',
    'checkpoints': 'checkpoints', 
    'sam_cache': 'sam_cache',
    'clipseg_cache': 'clipseg_cache',
    'visualizations': f'vis_pretrained_dlv3plus_cce_potts_{DISTANCE_TRANSFORM}'
}
for dir_name, dir_path in DIRS.items():
    full_path = os.path.join(DIRS['output'], dir_path) if dir_name != 'output' else dir_path
    os.makedirs(full_path, exist_ok=True)
PATHS = {
    'model_checkpoint': os.path.join(DIRS['output'], DIRS['checkpoints'], 'cce.pt'),
    'model': os.path.join(DIRS['output'], DIRS['checkpoints'], f'cce_potts_{DISTANCE_TRANSFORM}.pt'),
    'clipseg_pseudolabels': os.path.join(DIRS['output'], DIRS['clipseg_cache'], 'pseudolabels.npy'),
    'sam_contours_x': os.path.join(DIRS['output'], DIRS['sam_cache'], 'contours_x.npy'),
    'sam_contours_y': os.path.join(DIRS['output'], DIRS['sam_cache'], 'contours_y.npy'),
    'sam_checkpoint': os.path.join('sam_checkpoint', 'sam_vit_h_4b8939.pth')
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def generate_pseudolabels(voc_train_dataset):
    clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    pseudolabel_logits_arr = []
    for image, target in tqdm(voc_train_dataset):
        tags = [VOC_CLASSES[i] for i in np.unique(target)]
        if "ignore" in tags:
            tags.remove("ignore")
        inputs = clipseg_processor(text=tags, images=[image] * len(tags), padding="max_length", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        clipseg_model.eval()
        with torch.no_grad():
            outputs = clipseg_model(**inputs)

        preds = outputs.logits # preds have size (# tags, 352, 352)
        
        full_preds = torch.full((NUM_CLASSES, RESIZE_SIZE, RESIZE_SIZE), -1e9, dtype=preds.dtype, device=device)
        for i, tag in enumerate(tags):
            voc_id = VOC_CLASSES_FLIPPED.get(tag)
            full_preds[voc_id] = preds[i]

        pseudolabel_logits_arr.append(full_preds.detach().cpu().numpy())
    np.save(PATHS['clipseg_pseudolabels'], pseudolabel_logits_arr)
    print("All pseudolabels generated.")

def generate_sam_contours(voc_train_dataset):
    sam_checkpoint = PATHS['sam_checkpoint']
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    all_contours_x, all_contours_y = [], []
    for image, target in tqdm(voc_train_dataset):
        resized_image = np.array(image.resize((RESIZE_SIZE, RESIZE_SIZE), Image.BILINEAR))
        masks = mask_generator.generate(resized_image)
        contours_x = np.zeros((RESIZE_SIZE, RESIZE_SIZE - 1), dtype=bool)
        contours_y = np.zeros((RESIZE_SIZE - 1, RESIZE_SIZE), dtype=bool)

        for mask in masks:
            segmentation = mask['segmentation']
            contours_x |= np.logical_xor(segmentation[:, :-1], segmentation[:, 1:]) # shape: (H, W-1)
            contours_y |= np.logical_xor(segmentation[:-1, :], segmentation[1:, :]) # shape: (H-1, W)

        all_contours_x.append(contours_x)
        all_contours_y.append(contours_y)
    np.save(PATHS['sam_contours_x'], np.array(all_contours_x))
    np.save(PATHS['sam_contours_y'], np.array(all_contours_y))
    print("All SAM contours generated.")

class CustomVOCSegmentation(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        ])
        self.pseudolabel_logits_arr = np.load(PATHS['clipseg_pseudolabels'])
        self.sam_contours_x_arr = np.load(PATHS['sam_contours_x'])
        self.sam_contours_y_arr = np.load(PATHS['sam_contours_y'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        transformed_image = self.transform(image)
        pseudolabel_logits = torch.from_numpy(self.pseudolabel_logits_arr[idx])
        sam_contours_x = transforms.ToTensor()(self.sam_contours_x_arr[idx]).squeeze()
        sam_contours_y = transforms.ToTensor()(self.sam_contours_y_arr[idx]).squeeze()

        return transformed_image, pseudolabel_logits, sam_contours_x, sam_contours_y

def main():
    voc_train_dataset = datasets.VOCSegmentation(
        '.',
        image_set='train',
        download=False
    )

    if len(voc_train_dataset) > NUM_TRAIN_IMAGES:
        voc_train_dataset = Subset(voc_train_dataset, range(NUM_TRAIN_IMAGES))
        print(f"Limiting training to the first {NUM_TRAIN_IMAGES} images.")

    # generate_pseudolabels(voc_train_dataset)
    # generate_sam_contours(voc_train_dataset)
    train_dataset = CustomVOCSegmentation(voc_train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    deeplabv3plus = deeplabv3plus_resnet101().to(device)
    optimizer = optim.Adam(deeplabv3plus.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    if os.path.exists(PATHS['model_checkpoint']):
        print(f"Loading checkpoint from {PATHS['model_checkpoint']}...")
        checkpoint = torch.load(PATHS['model_checkpoint'], map_location=device)
        deeplabv3plus.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming training")
    else:
        print("No checkpoint found, starting training from epoch 0.")

    print("\nStarting training...")
    epoch_total_losses = []
    epoch_cce_main_losses = []
    epoch_potts_main_losses = []
    for epoch in range(start_epoch, NUM_EPOCHS):
        deeplabv3plus.train()
        
        running_total_loss = 0.0
        running_cce_main = 0.0
        running_potts_main = 0.0
        for i, (transformed_images, pseudolabel_logits_batch, sam_contours_x_batch, sam_contours_y_batch) in enumerate(train_loader):
            transformed_images = transformed_images.to(device)
            pseudolabel_logits_batch = pseudolabel_logits_batch.to(device)
            sam_contours_x_batch = sam_contours_x_batch.to(device)
            sam_contours_y_batch = sam_contours_y_batch.to(device)

            optimizer.zero_grad()
            outputs = deeplabv3plus(transformed_images)
            
            # unary potential
            cce_loss_main = CollisionCrossEntropyLoss(outputs, pseudolabel_logits_batch)

            # pairwise potential
            potts_loss_main = BLPottsLoss(outputs, sam_contours_x_batch, sam_contours_y_batch, distance_transform=DISTANCE_TRANSFORM) # torch.tensor(0.0, device=device)

            total_loss = cce_loss_main + potts_loss_main

            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item()
            running_cce_main += cce_loss_main.item()
            running_potts_main += potts_loss_main.item()
        
        num_batches = len(train_loader)
        loss_data = [
            (running_total_loss, epoch_total_losses),
            (running_cce_main, epoch_cce_main_losses),
            (running_potts_main, epoch_potts_main_losses)
        ]
        for running_loss_sum, epoch_loss_list in loss_data:
            avg_loss = running_loss_sum / num_batches
            epoch_loss_list.append(avg_loss)
            
        print(f"Epoch {epoch+1} finished. "
            f"Average Total Loss: {epoch_total_losses[-1]:.4f}, "
            f"Avg CCE Main: {epoch_cce_main_losses[-1]:.4f}, "
            f"Avg Potts Main: {epoch_potts_main_losses[-1]:.4f}"
            )

    print("\nTraining complete!")

    torch.save({
        'epoch': epoch,
        'model_state_dict': deeplabv3plus.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_losses': epoch_total_losses,
        'cce_main_losses': epoch_cce_main_losses,
        'potts_main_losses': epoch_potts_main_losses
    }, PATHS['model'])
    print(f"Checkpoint saved to {PATHS['model']}")

    # generate visualizations for all training images
    vis_output_dir = os.path.join(DIRS['output'], DIRS['visualizations'])
    for i in range(NUM_TRAIN_IMAGES):
        vis_sample_img(voc_train_dataset, train_dataset, deeplabv3plus, i, DISTANCE_TRANSFORM, vis_output_dir)
    vis_training_loss(NUM_EPOCHS, epoch_total_losses, epoch_cce_main_losses, epoch_potts_main_losses, vis_output_dir)

if __name__ == "__main__":
    main()