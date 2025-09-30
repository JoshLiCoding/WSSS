import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from model.deeplab import deeplabv3plus_resnet101
from model.scheduler import PolyLR
from utils.dataset import VOCSegmentation, CustomVOCSegmentationTrain, CustomVOCSegmentationVal
from utils.loss import CrossEntropyLoss, CollisionCrossEntropyLoss, BLPottsLoss
from utils.metrics import update_miou
from vis import vis_train_sample_img, vis_val_sample_img, vis_train_loss, vis_val_loss

VOC_CLASSES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}
VOC_CLASSES_FLIPPED = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15, "potted plant": 16, "sheep": 17, "sofa": 18, "train": 19, "tv/monitor": 20, "ignore": 255}

# NUM_TRAIN_IMAGES = 200
NUM_CLASSES = 21
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
IGNORE_INDEX = 255
RESIZE_SIZE = 352
VALIDATION_INTERVAL = 10
DISTANCE_TRANSFORM = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIRS = {
    'output': 'output',
    'checkpoints': 'checkpoints', 
    'sam_cache': 'sam_cache',
    'clipseg_cache': 'clipseg_cache',
    'visualizations': f'vis_{NUM_EPOCHS}epochs_ce_init_1'
}
for dir_name, dir_path in DIRS.items():
    full_path = os.path.join(DIRS['output'], dir_path) if dir_name != 'output' else dir_path
    os.makedirs(full_path, exist_ok=True)
PATHS = {
    'model_checkpoint': os.path.join(DIRS['output'], DIRS['checkpoints'], 'none.pt'),
    'model': os.path.join(DIRS['output'], DIRS['checkpoints'], f'ce_1.pt'),
    'sam_contours_x': os.path.join(DIRS['output'], DIRS['sam_cache'], 'contours_x_aug.npy'),
    'sam_contours_y': os.path.join(DIRS['output'], DIRS['sam_cache'], 'contours_y_aug.npy'),
    'sam_checkpoint': os.path.join('sam_checkpoint', 'sam_vit_h_4b8939.pth')
}

def generate_pseudolabels(voc_train_dataset):
    clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    for idx, (image, target) in enumerate(tqdm(voc_train_dataset, desc="Generating CLIPSeg pseudolabels")):
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

        # Save each pseudolabel as a separate .npy file
        np.save(os.path.join(DIRS['clipseg_cache'], f"pseudolabel_{idx}.npy"), full_preds.detach().cpu().numpy())
    print(f"All pseudolabels saved individually to {DIRS['clipseg_cache']}.")

def generate_sam_contours(voc_train_dataset):
    sam_checkpoint = PATHS['sam_checkpoint']
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    all_contours_x, all_contours_y = [], []
    for image, target in tqdm(voc_train_dataset, desc="Generating SAM contours"):
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

def main():
    # augmented VOC train set
    voc_train_dataset = VOCSegmentation(
        '.',
        image_set='train',
        download=False
    )
    voc_val_dataset = VOCSegmentation(
        '.',
        image_set='val',
        download=False
    )

    # if len(voc_train_dataset) > NUM_TRAIN_IMAGES:
    #     voc_train_dataset = Subset(voc_train_dataset, range(NUM_TRAIN_IMAGES))
    #     print(f"Limiting training to the first {NUM_TRAIN_IMAGES} images.")
    print(f"Training on {len(voc_train_dataset)} images.")

    generate_pseudolabels(voc_train_dataset)
    # generate_sam_contours(voc_train_dataset)

    train_dataset = CustomVOCSegmentationTrain(voc_train_dataset, PATHS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_dataset = CustomVOCSegmentationVal(voc_val_dataset)

    deeplabv3plus = deeplabv3plus_resnet101().to(device)
    optimizer = torch.optim.SGD(params=[
        {'params': deeplabv3plus.backbone.parameters(), 'lr': 0.1 * LEARNING_RATE},
        {'params': deeplabv3plus.classifier.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = PolyLR(optimizer, NUM_EPOCHS * len(train_loader), power=0.9)

    if os.path.exists(PATHS['model_checkpoint']):
        print(f"Loading checkpoint from {PATHS['model_checkpoint']}...")
        checkpoint = torch.load(PATHS['model_checkpoint'], map_location=device)
        deeplabv3plus.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming training")
    else:
        print("No checkpoint found, starting training from epoch 0.")

    print("\nStarting training...")
    epoch_total_losses = []
    epoch_cce_main_losses = []
    epoch_potts_main_losses = []
    validation_mious = []
    validation_epochs = []
    best_miou = 0.0
    best_epoch = 0
    
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training epochs"):
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
            cce_loss_main = CollisionCrossEntropyLoss(outputs, pseudolabel_logits_batch) # CrossEntropyLoss(outputs, pseudolabel_logits_batch)
            # pairwise potential
            potts_loss_main = torch.tensor(0.0, device=device) # BLPottsLoss(outputs, sam_contours_x_batch, sam_contours_y_batch, distance_transform=DISTANCE_TRANSFORM)

            total_loss = cce_loss_main + potts_loss_main

            total_loss.backward()
            optimizer.step()
            scheduler.step()

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
        
        # Validation
        if (epoch + 1) % VALIDATION_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Running validation at epoch {epoch + 1}...")
            deeplabv3plus.eval()
            
            # Initialize per-class intersection and union counters
            intersection_counts = np.zeros(NUM_CLASSES)
            union_counts = np.zeros(NUM_CLASSES)
            
            with torch.no_grad():
                for val_transformed_image, val_target in val_dataset:
                    val_transformed_image = val_transformed_image.to(device)
                    val_target = val_target.to(device)

                    val_outputs = deeplabv3plus(val_transformed_image.unsqueeze(0))
                    update_miou(val_outputs, val_target.unsqueeze(0), intersection_counts, union_counts, NUM_CLASSES, IGNORE_INDEX)

            ious = []
            for cls in range(NUM_CLASSES):
                if cls == IGNORE_INDEX:
                    continue
                if union_counts[cls] == 0:
                    continue  # Skip classes not present in dataset
                else:
                    iou = intersection_counts[cls] / union_counts[cls]
                    ious.append(iou)
            avg_miou = np.mean(ious)
            validation_mious.append(avg_miou)
            validation_epochs.append(epoch + 1)
            
            print(f"Validation mIoU: {avg_miou:.4f}")
            
            # Save best model based on validation mIoU
            if avg_miou > best_miou:
                best_miou = avg_miou
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': deeplabv3plus.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'total_loss': epoch_total_losses[-1],
                    'cce_main_loss': epoch_cce_main_losses[-1],
                    'potts_main_loss': epoch_potts_main_losses[-1],
                    'validation_miou': avg_miou
                }, PATHS['model'])
                print(f"New best model saved! mIoU: {best_miou:.4f} at epoch {best_epoch}")

    print(f"\nTraining complete! Best model was at epoch {best_epoch} with mIoU {best_miou:.4f}")
    
    # Load best model for inference/visualization
    if os.path.exists(PATHS['model']):
        best_checkpoint = torch.load(PATHS['model'], map_location=device, weights_only=False)
        deeplabv3plus.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Best model loaded successfully! Final validation mIoU: {best_miou:.4f}")
    
    # generate visualizations for all training images
    vis_output_dir = os.path.join(DIRS['output'], DIRS['visualizations'])
    for i in range(0, len(voc_train_dataset), 200):
        vis_train_sample_img(voc_train_dataset, train_dataset, deeplabv3plus, i, DISTANCE_TRANSFORM, vis_output_dir)
    for i in range(0, len(voc_val_dataset), 50):
        vis_val_sample_img(voc_val_dataset, val_dataset, deeplabv3plus, i, vis_output_dir)
    vis_train_loss(NUM_EPOCHS, epoch_total_losses, epoch_cce_main_losses, epoch_potts_main_losses, vis_output_dir)
    vis_val_loss(NUM_EPOCHS, validation_mious, validation_epochs, vis_output_dir)

if __name__ == "__main__":
    main()