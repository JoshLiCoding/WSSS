import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from model.dino import DinoWSSS
from model.scheduler import PolyLR
from model.dino_txt import generate_pseudolabels
from utils.dataset import VOCSegmentation, CustomVOCSegmentationTrain, CustomVOCSegmentationVal
from utils.loss import CrossEntropyLoss, CollisionCrossEntropyLoss, PottsLoss
from utils.metrics import update_miou
from vis import vis_train_sample_img, vis_val_sample_img, vis_train_loss, vis_val_loss

NUM_CLASSES = 21
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
IGNORE_INDEX = 255
VALIDATION_INTERVAL = 10
POTTS_TYPE = 'quadratic'
DISTANCE_TRANSFORM = None
TRAIN_ONLY = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIRS = {
    'output': 'output',
    'checkpoints': 'checkpoints',
    'sam_cache': 'sam_cache',
    'visualizations': f'vis_{NUM_EPOCHS}epochs_dino_wsss'
}
for dir_name, dir_path in DIRS.items():
    full_path = os.path.join(DIRS['output'], dir_path) if dir_name != 'output' else dir_path
    os.makedirs(full_path, exist_ok=True)
PATHS = {
    'model_checkpoint': os.path.join(DIRS['output'], DIRS['checkpoints'], 'none.pt'),
    'model': os.path.join(DIRS['output'], DIRS['checkpoints'], f'dino_wsss_10epochs.pt'),
    'sam_checkpoint': os.path.join('sam_checkpoint', 'sam_vit_h_4b8939.pth')
}

def generate_sam_contours(voc_train_dataset):
    sam_checkpoint = PATHS['sam_checkpoint']
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    for i, (image, target) in tqdm(enumerate(voc_train_dataset), desc="Generating SAM contours"):
        image = np.array(image)
        masks = mask_generator.generate(image)
        H, W = image.shape[:2]
        contours_x = np.zeros((H, W - 1), dtype=bool)
        contours_y = np.zeros((H - 1, W), dtype=bool)

        for mask in masks:
            segmentation = mask['segmentation']
            contours_x |= np.logical_xor(segmentation[:, :-1], segmentation[:, 1:]) # shape: (H, W-1)
            contours_y |= np.logical_xor(segmentation[:-1, :], segmentation[1:, :]) # shape: (H-1, W)
        
        np.save(os.path.join(DIRS['output'], DIRS['sam_cache'], f'sam_contours_x_{i}.npy'), contours_x)
        np.save(os.path.join(DIRS['output'], DIRS['sam_cache'], f'sam_contours_y_{i}.npy'), contours_y)

    print("All SAM contours generated.")

def main():
    # augmented VOC train set
    voc_train_dataset = VOCSegmentation(
        '.',
        image_set='train',
        download=False,
        n_images=5700
    )
    voc_val_dataset = VOCSegmentation(
        '.',
        image_set='val',
        download=False
    )
    print(f"Training on {len(voc_train_dataset)} images.")

    # generate_sam_contours(voc_train_dataset)
    # generate_pseudolabels(voc_train_dataset, start_index=5700)

    train_dataset = CustomVOCSegmentationTrain(
        voc_train_dataset, NUM_CLASSES, 
        os.path.join(DIRS['output'], DIRS['sam_cache']),
        'pseudolabels'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_dataset = CustomVOCSegmentationVal(voc_val_dataset)

    model = DinoWSSS(
        backbone_name="dinov3_vitl16",
        num_transformer_blocks=2,
        num_conv_blocks=3,
        out_channels=21
    ).to(device)
    optimizer = torch.optim.SGD(params=[
        {'params': model.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    # scheduler = PolyLR(optimizer, NUM_EPOCHS * len(train_loader), power=0.9)

    if os.path.exists(PATHS['model_checkpoint']):
        print(f"Loading checkpoint from {PATHS['model_checkpoint']}...")
        checkpoint = torch.load(PATHS['model_checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resuming training")
    else:
        print("No checkpoint found, starting training from epoch 0.")

    print("\nStarting training...")
    epoch_total_losses = []
    epoch_unary_losses = []
    epoch_pairwise_losses = []
    validation_mious = []
    validation_epochs = []
    best_miou = 0.0
    best_epoch = 0
    
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training epochs"):
        model.train()
        
        running_total_loss = 0.0
        running_unary_loss = 0.0
        running_pairwise_loss = 0.0
        for i, (transformed_images, pseudolabel_logits, sam_contours_x_batch, sam_contours_y_batch) in enumerate(train_loader):
            transformed_images = transformed_images.to(device)
            pseudolabel_logits = pseudolabel_logits.to(device)
            sam_contours_x_batch = sam_contours_x_batch.to(device)
            sam_contours_y_batch = sam_contours_y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(transformed_images)

            # unary potential
            unary_loss = CollisionCrossEntropyLoss(outputs, pseudolabel_logits) # torch.tensor(0.0, device=device)

            # pairwise potential
            pairwise_loss = torch.tensor(0.0, device=device) # PottsLoss(POTTS_TYPE, outputs, sam_contours_x_batch, sam_contours_y_batch, DISTANCE_TRANSFORM)

            total_loss = unary_loss + pairwise_loss

            total_loss.backward()
            optimizer.step()
            # scheduler.step()

            running_total_loss += total_loss.item()
            running_unary_loss += unary_loss.item()
            running_pairwise_loss += pairwise_loss.item()

            if epoch == 0 and i == 0:
                print(f"Initial losses -- Total: {total_loss.item():.4f}, Unary: {unary_loss.item():.4f}, Pairwise: {pairwise_loss.item():.4f}")

        num_batches = len(train_loader)
        loss_data = [
            (running_total_loss, epoch_total_losses),
            (running_unary_loss, epoch_unary_losses),
            (running_pairwise_loss, epoch_pairwise_losses)
        ]
        for running_loss_sum, epoch_loss_list in loss_data:
            avg_loss = running_loss_sum / num_batches
            epoch_loss_list.append(avg_loss)
            
        print(f"Epoch {epoch+1} finished. "
            f"Average Total Loss: {epoch_total_losses[-1]:.4f}, "
            f"Avg Unary: {epoch_unary_losses[-1]:.4f}, "
            f"Avg Pairwise: {epoch_pairwise_losses[-1]:.4f}"
            )
        
        # validation
        if (epoch + 1) % VALIDATION_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            if TRAIN_ONLY:
                print("TRAIN_ONLY is set to True, skipping validation. Saving model checkpoint...")
                torch.save({
                    'model_state_dict': model.state_dict()
                }, PATHS['model'])
                continue
            
            print(f"Running validation at epoch {epoch + 1}...")
            model.eval()
            
            # initialize per-class intersection and union counters
            intersection_counts = np.zeros(NUM_CLASSES)
            union_counts = np.zeros(NUM_CLASSES)
            
            with torch.no_grad():
                for val_transformed_image, val_target in val_dataset:
                    val_transformed_image = val_transformed_image.to(device)
                    val_target = val_target.to(device)

                    val_outputs = model(val_transformed_image.unsqueeze(0))
                    update_miou(val_outputs, val_target.unsqueeze(0), intersection_counts, union_counts, NUM_CLASSES, IGNORE_INDEX)

            ious = []
            for cls in range(NUM_CLASSES):
                if cls == IGNORE_INDEX:
                    continue
                if union_counts[cls] == 0:
                    continue
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
                    'model_state_dict': model.state_dict()
                }, PATHS['model'])
                print(f"New best model saved! mIoU: {best_miou:.4f} at epoch {best_epoch}")

    print(f"\nTraining complete! Best model was at epoch {best_epoch} with mIoU {best_miou:.4f}")
    
    if os.path.exists(PATHS['model']):
        best_checkpoint = torch.load(PATHS['model'], map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Best model loaded successfully! Final validation mIoU: {best_miou:.4f}")
    
    vis_output_dir = os.path.join(DIRS['output'], DIRS['visualizations'])
    for i in range(0, len(voc_train_dataset), 100):
        vis_train_sample_img(voc_train_dataset, train_dataset, model, i, DISTANCE_TRANSFORM, vis_output_dir)
    vis_train_loss(NUM_EPOCHS, epoch_total_losses, epoch_unary_losses, epoch_pairwise_losses, vis_output_dir)
    
    if not TRAIN_ONLY:
        for i in range(0, len(voc_val_dataset), 50):
            vis_val_sample_img(voc_val_dataset, val_dataset, model, i, vis_output_dir)
        vis_val_loss(validation_mious, validation_epochs, vis_output_dir)

if __name__ == "__main__":
    main()