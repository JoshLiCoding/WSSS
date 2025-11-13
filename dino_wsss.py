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
import yaml
import wandb
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from model.dino import DinoWSSS
from model.deeplab import deeplabv3_resnet101, deeplabv3plus_resnet101
from model.scheduler import PolyLR
from model.dino_txt_full_img import generate_pseudolabels
from utils.dataset import VOCSegmentation, COCOSegmentation, CustomSegmentationTrain, CustomSegmentationVal
from utils.loss import CollisionCrossEntropyLoss, PottsLoss
from utils.metrics import update_miou
from vis import vis_train_sample_img, vis_val_sample_img, vis_train_loss, vis_val_loss

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def print_config(config):
    """Print configuration parameters in a readable format"""
    print("=" * 60)
    print("CONFIGURATION PARAMETERS")
    print("=" * 60)
    
    for section_name, section_params in config.items():
        print(f"\n{section_name.upper()}:")
        print("-" * 30)
        if isinstance(section_params, dict):
            for key, value in section_params.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {section_params}")
    
    print("=" * 60)

# Load configuration
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract parameters from config
NUM_CLASSES = config['model']['num_classes']
BATCH_SIZE = config['training']['batch_size']
NUM_EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
WEIGHT_DECAY = config['training']['weight_decay']
MOMENTUM = config['training']['momentum']
IGNORE_INDEX = config['training']['ignore_index']
VALIDATION_INTERVAL = config['training']['validation_interval']
POTTS_TYPE = config['loss']['potts_type']
DISTANCE_TRANSFORM = config['loss']['distance_transform']
TRAIN_ONLY = config['training']['train_only']
CLASS_NAMES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}

# Setup directories and paths
DIRS = {
    'output': config['directories']['output'],
    'checkpoints': config['directories']['checkpoints'],
    'visualizations': config['directories']['visualizations'].format(num_epochs=NUM_EPOCHS)
}

# Create directories
for dir_name, dir_path in DIRS.items():
    full_path = os.path.join(DIRS['output'], dir_path) if dir_name != 'output' else dir_path
    os.makedirs(full_path, exist_ok=True)

PATHS = {
    'model_checkpoint': config['paths']['model_checkpoint'],
    'model': config['paths']['model'].format(num_epochs=NUM_EPOCHS),
    'sam_checkpoint': config['sam']['checkpoint_path']
}

def generate_sam_contours(voc_train_dataset, start_index=0):
    sam_checkpoint = PATHS['sam_checkpoint']
    model_type = config['sam']['model_type']
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    for i, (image, target) in tqdm(enumerate(voc_train_dataset), desc="Generating SAM contours"):
        if i < start_index:
            continue
        image = np.array(image)
        masks = mask_generator.generate(image)
        H, W = image.shape[:2]
        contours_x = np.zeros((H, W - 1), dtype=bool)
        contours_y = np.zeros((H - 1, W), dtype=bool)

        for mask in masks:
            segmentation = mask['segmentation']
            contours_x |= np.logical_xor(segmentation[:, :-1], segmentation[:, 1:]) # shape: (H, W-1)
            contours_y |= np.logical_xor(segmentation[:-1, :], segmentation[1:, :]) # shape: (H-1, W)

        np.save(os.path.join(config['directories']['sam_cache'], f'sam_contours_x_{i}.npy'), contours_x)
        np.save(os.path.join(config['directories']['sam_cache'], f'sam_contours_y_{i}.npy'), contours_y)

    print("All SAM contours generated.")

def main():
    # Print configuration parameters
    print_config(config)
    
    # Initialize Weights & Biases
    wandb_config = config['wandb']
    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config['entity'],
        name=DIRS['visualizations'],
        config=config  # Log the entire configuration
    )
    
    # augmented VOC train set
    if config['dataset']['dataset_name'] == 'voc':
        original_train_dataset = VOCSegmentation(
            config['dataset']['root'],
            image_set=config['dataset']['train_image_set'],
            download=config['dataset']['download'],
            n_images=config['dataset']['n_images']
    )
    elif config['dataset']['dataset_name'] == 'coco':
        original_train_dataset = COCOSegmentation(
            os.path.join(config['dataset']['root'], 'coco'),
            image_set=config['dataset']['train_image_set'],
            download=config['dataset']['download'],
            n_images=config['dataset']['n_images']
        )

    if config['dataset']['dataset_name'] == 'voc':
        original_val_dataset = VOCSegmentation(
            config['dataset']['root'],
            image_set=config['dataset']['val_image_set'],
            download=config['dataset']['download']
        )
    elif config['dataset']['dataset_name'] == 'coco':
        original_val_dataset = COCOSegmentation(
            os.path.join(config['dataset']['root'], 'coco'),
            image_set=config['dataset']['val_image_set'],
            download=config['dataset']['download']
        )
    
    # generate_sam_contours(original_train_dataset)
    generate_pseudolabels(original_train_dataset, config['dataset']['dataset_name'])
    # 9500, 19000
    return

    train_dataset = CustomSegmentationTrain(
        original_train_dataset, NUM_CLASSES, 
        config['directories']['sam_cache'],
        config['directories']['pseudolabels']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_dataset = CustomSegmentationVal(original_val_dataset)
    model = DinoWSSS(
        backbone_name=config['model']['backbone_name'],
        num_transformer_blocks=config['model']['num_transformer_blocks'],
        num_conv_blocks=config['model']['num_conv_blocks'],
        out_channels=config['model']['out_channels'],
        use_bottleneck=config['model']['use_bottleneck']
    ).to(device)
    optimizer = torch.optim.SGD(params=[
        {'params': model.transformer_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.conv_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.conv_final.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # scheduler = PolyLR(optimizer, NUM_EPOCHS * len(train_loader), power=0.9)

    # model = deeplabv3_resnet101(NUM_CLASSES).to(device)
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.backbone.parameters(), 'lr': LEARNING_RATE},
    #     {'params': model.classifier.parameters(), 'lr': LEARNING_RATE},
    # ], lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
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
        for i, (transformed_images, targets, pseudolabel_probs, sam_contours_x_batch, sam_contours_y_batch) in enumerate(train_loader):
            transformed_images = transformed_images.to(device)
            targets = targets.to(device)
            pseudolabel_probs = pseudolabel_probs.to(device)
            sam_contours_x_batch = sam_contours_x_batch.to(device)
            sam_contours_y_batch = sam_contours_y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(transformed_images)

            # unary potential
            unary_loss = CollisionCrossEntropyLoss(outputs, pseudolabel_probs) # torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(outputs, targets)

            # pairwise potential
            pairwise_loss = torch.tensor(0.0, device=device) # PottsLoss(POTTS_TYPE, outputs, sam_contours_x_batch, sam_contours_y_batch, DISTANCE_TRANSFORM) # torch.tensor(0.0, device=device)

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
        
        # Log training losses to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/total_loss": epoch_total_losses[-1],
            "train/unary_loss": epoch_unary_losses[-1],
            "train/pairwise_loss": epoch_pairwise_losses[-1]
        })
        
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
                    print(f"Class {CLASS_NAMES[cls]} mIoU: {iou:.4f}")
            avg_miou = np.mean(ious)
            validation_mious.append(avg_miou)
            validation_epochs.append(epoch + 1)
            
            print(f"Validation mIoU: {avg_miou:.4f}")
            
            # Log validation mIoU to wandb
            wandb.log({
                "epoch": epoch + 1,
                "val/miou": avg_miou,
                "val/best_miou": best_miou
            })
            
            # Save best model based on validation mIoU
            if avg_miou > best_miou:
                best_miou = avg_miou
                best_epoch = epoch + 1
                torch.save({
                    'model_state_dict': model.state_dict()
                }, PATHS['model'])
                print(f"New best model saved! mIoU: {best_miou:.4f} at epoch {best_epoch}")
                

    print(f"\nTraining complete! Best model was at epoch {best_epoch} with mIoU {best_miou:.4f}")
    
    # Log final summary to wandb
    wandb.log({
        "final/best_miou": best_miou,
        "final/best_epoch": best_epoch,
        "final/total_epochs": NUM_EPOCHS
    })
    
    if os.path.exists(PATHS['model']):
        best_checkpoint = torch.load(PATHS['model'], map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Best model loaded successfully! Final validation mIoU: {best_miou:.4f}")
    
    vis_output_dir = os.path.join(DIRS['output'], DIRS['visualizations'])
    for i in range(0, len(original_train_dataset), config['visualization']['train_sample_interval']):
        vis_train_sample_img(original_train_dataset, train_dataset, model, i, DISTANCE_TRANSFORM, vis_output_dir)
    vis_train_loss(NUM_EPOCHS, epoch_total_losses, epoch_unary_losses, epoch_pairwise_losses, vis_output_dir)
    
    if not TRAIN_ONLY:
        for i in range(0, len(original_val_dataset), config['visualization']['val_sample_interval']):
            vis_val_sample_img(original_val_dataset, val_dataset, model, i, vis_output_dir)
        vis_val_loss(validation_mious, validation_epochs, vis_output_dir)
    
    # Log visualizations to wandb
    if wandb_config['log_visualizations']:
        for file in os.listdir(vis_output_dir):
            if file.endswith('.png'):
                wandb.log({"plots/visualizations": wandb.Image(os.path.join(vis_output_dir, file))})
        
    wandb.finish()

if __name__ == "__main__":
    main()