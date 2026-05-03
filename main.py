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
from ultralytics import FastSAM
import ultralytics
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import clip

from model.clip import ClipWSSS
from model.deeplab import deeplabv3_resnet101, deeplabv3plus_resnet101
from model.scheduler import PolyLR
from model.dino_txt_full_img import generate_pseudolabels_batch, build_text_embeddings, get_class_names_from_config
from utils.dataset import VOCSegmentation, COCOSegmentation, CustomSegmentationTrain, CustomSegmentationVal
from utils.loss import CollisionCrossEntropyLoss, PottsLoss, KLDivergenceLoss, CrossEntropyLoss, ReverseCrossEntropyLoss, KLEntropyLoss
from utils.metrics import update_miou
from utils.sam import (
    generate_sam_contours_batch,
    generate_fastsam_contours_batch,
    generate_slic_contours_batch,
    generate_gt_contours_batch,
    generate_color_diff_contours_batch,
)
from utils.vis import vis_train_sample_img, vis_val_sample_img, vis_train_loss, vis_val_loss


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
CONTOUR_METHOD = config['loss']['contour_method']
TRAIN_ONLY = config['training']['train_only']

VOC_CLASS_NAMES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}

COCO_CLASS_NAMES = {0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat", 36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle", 41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 80: "toothbrush", 255: "ignore"}

CLASS_NAMES = VOC_CLASS_NAMES if config['dataset']['dataset_name'] == 'voc' else COCO_CLASS_NAMES

# Setup directories and paths
DIRS = {
    'output': config['directories']['output'],
    'checkpoints': config['directories']['checkpoints'],
    'visualizations': config['directories']['visualizations'].format(num_epochs=NUM_EPOCHS)
}
for dir_name, dir_path in DIRS.items():
    full_path = os.path.join(DIRS['output'], dir_path) if dir_name != 'output' else dir_path
    os.makedirs(full_path, exist_ok=True)
PATHS = {
    'model': config['paths']['model'].format(num_epochs=NUM_EPOCHS),
    'sam_checkpoint': config['paths']['sam_checkpoint']
}

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

    train_dataset = CustomSegmentationTrain(
        original_train_dataset
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_dataset = CustomSegmentationVal(original_val_dataset)
    
    # Initialize FastSAM model for batch processing
    fastsam_model = FastSAM('FastSAM-x.pt')
    
    # Initialize SAM model for automatic mask generation (if needed)
    sam_model = None
    sam_mask_generator = None
    if CONTOUR_METHOD == 'sam':
        sam_checkpoint = PATHS['sam_checkpoint']
        model_type = "vit_b"  # or "vit_b", "vit_l" depending on checkpoint
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=16,
            points_per_batch=256,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8
        )
        print("SAM model initialized for automatic mask generation")
    
    clip_model, _ = clip.load(config["model"]["clip_model_name"], device=device, jit=False)
    clip_model = clip_model.float()
    clip_model.eval()

    # Precompute text embeddings for all classes (once before training)
    print("Precomputing text embeddings for all classes...")
    all_fg_class_names, background_class_names, num_all_fg, num_bg = get_class_names_from_config(config)
    all_class_names = all_fg_class_names + background_class_names
    text_emb_all = build_text_embeddings(clip_model, all_class_names, device=device)
    print(f"Text embeddings computed: shape {text_emb_all.shape}")
    
    model = ClipWSSS(
        clip_model,
        num_transformer_blocks=config["model"]["num_transformer_blocks"],
        num_conv_blocks=config["model"]["num_conv_blocks"],
        out_channels=config["model"]["out_channels"],
        use_bottleneck=config["model"]["use_bottleneck"],
        use_transpose_conv=config["model"]["use_transpose_conv"],
    ).to(device)
    model.visual.eval()

    optimizer_params = [
        {'params': model.transformer_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.ln.parameters(), 'lr': LEARNING_RATE},
        {'params': model.conv_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.lin_classifier.parameters(), 'lr': LEARNING_RATE},
    ]
    if config['model']['use_transpose_conv']:
        optimizer_params.append({'params': model.upsample_conv1.parameters(), 'lr': LEARNING_RATE})
        optimizer_params.append({'params': model.upsample_conv2.parameters(), 'lr': LEARNING_RATE})
    
    optimizer = torch.optim.SGD(
        params=optimizer_params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    max_iters = NUM_EPOCHS * len(train_loader)
    # scheduler = PolyLR(optimizer, max_iters=max_iters)

    model_checkpoint = config['paths']['model_checkpoint']
    if os.path.exists(model_checkpoint):
        print(f"Loading checkpoint from {model_checkpoint}...")
        checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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
        model.visual.eval()
        
        running_total_loss = 0.0
        running_unary_loss = 0.0
        running_pairwise_loss = 0.0
        for i, (transformed_images, targets) in enumerate(train_loader):
            transformed_images = transformed_images.to(device)

            optimizer.zero_grad()
            
            model_outputs = model(transformed_images)
            segmentations = model_outputs["seg"]
            clip_patch_tokens = model_outputs["clip"]

            with torch.no_grad():
                pseudolabels_batch, class_indices_batch = generate_pseudolabels_batch(
                    clip_patch_tokens, targets, text_emb_all, num_all_fg, num_bg
                )
            
            # Convert pseudolabels to tensor format matching segmentation output
            # segmentations shape: [B, C, H, W] where C=21 for VOC
            B, _, H_seg, W_seg = segmentations.shape
            pseudolabel_probs = torch.zeros((B, NUM_CLASSES, H_seg, W_seg), dtype=torch.float32, device=device)
            
            for b in range(B):
                pseudolabel = pseudolabels_batch[b]  # [num_fg + 1, p, p] or [1, p, p] if no fg classes (C, H, W format)
                class_indices = class_indices_batch[b]
                
                # pseudolabel is already in [C, H, W] format, just add batch dimension
                pseudolabel_tensor = torch.from_numpy(pseudolabel).float()  # [num_fg+1, p, p] or [1, p, p] (C, H, W)
                pseudolabel_tensor = pseudolabel_tensor.unsqueeze(0)  # [1, num_fg+1, p, p] or [1, 1, p, p] (N, C, H, W)
                
                # Interpolate to segmentation size
                pseudolabel_tensor = F.interpolate(pseudolabel_tensor, size=(H_seg, W_seg), mode='bilinear', align_corners=False)
                pseudolabel_tensor = pseudolabel_tensor[0]  # [num_fg+1, H_seg, W_seg] or [1, H_seg, W_seg] (C, H, W)
                
                # Softmax with temperature
                t = 0.05
                pseudolabel_probs_b = torch.softmax(pseudolabel_tensor / t, dim=0)

                # Min-max normalize channel-wise, then renormalize to probability simplex
                min_vals = pseudolabel_probs_b.view(pseudolabel_probs_b.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
                max_vals = pseudolabel_probs_b.view(pseudolabel_probs_b.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
                pseudolabel_probs_b = (pseudolabel_probs_b - min_vals) / (max_vals - min_vals + 1e-8)

                pseudolabel_probs_b = pseudolabel_probs_b / (pseudolabel_probs_b.sum(dim=0, keepdim=True) + 1e-8)
                
                # [TEST] Dense CRF at 4x downsampled resolution (112x112) — remove this block after testing
                # import pydensecrf.densecrf as dcrf
                # from pydensecrf.utils import unary_from_softmax
                # from utils.dataset import MEAN, STD
                # probs_np = pseudolabel_probs_b.cpu().numpy().astype(np.float32)
                # Cc, Hc, Wc = probs_np.shape
                # img_112 = F.interpolate(transformed_images[b : b + 1], size=(Hc, Wc), mode='bilinear', align_corners=False)[0].cpu()
                # img_uint8 = np.ascontiguousarray((img_112.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1) * 255).astype(np.uint8)
                # d = dcrf.DenseCRF2D(Wc, Hc, Cc)
                # d.setUnaryEnergy(unary_from_softmax(np.ascontiguousarray(probs_np)))
                # d.addPairwiseBilateral(sxy=20, srgb=13, rgbim=img_uint8, compat=10)
                # d.addPairwiseGaussian(sxy=1, compat=3)
                # pseudolabel_probs_b = torch.from_numpy(np.array(d.inference(10)).reshape(Cc, Hc, Wc)).float().to(device)
                
                # Map to full class space [NUM_CLASSES, H_seg, W_seg]
                if len(class_indices) == 0:
                    # Only background class
                    pseudolabel_probs[b, 0] = pseudolabel_probs_b[0]  # background
                else:
                    for idx, class_idx in enumerate(class_indices):
                        pseudolabel_probs[b, class_idx] = pseudolabel_probs_b[idx]
                    pseudolabel_probs[b, 0] = pseudolabel_probs_b[len(class_indices)]  # background
            
            # Generate contours based on the selected method
            if CONTOUR_METHOD == 'gt':
                sam_contours_x_batch, sam_contours_y_batch = generate_gt_contours_batch(targets, device)
            elif CONTOUR_METHOD == 'color_diff':
                # Denormalize images for color difference computation
                images_denorm = []
                for img in transformed_images:
                    images_denorm.append(train_dataset.denormalize(img.clone()))
                images_denorm_batch = torch.stack(images_denorm).to(device)  # (B, C, H, W) in [0, 1]
                # Resize to segmentation size
                images_denorm_batch = F.interpolate(images_denorm_batch, size=(segmentations.shape[2], segmentations.shape[3]), mode='bilinear', align_corners=False)
                sam_contours_x_batch, sam_contours_y_batch = generate_color_diff_contours_batch(images_denorm_batch, device)
            elif CONTOUR_METHOD in ('sam', 'fastsam', 'slic'):
                def _to_pil(img_t):
                    img_denorm = train_dataset.denormalize(img_t.clone())
                    img_np = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    return Image.fromarray(img_np)

                images_pil = [_to_pil(img) for img in transformed_images]

                if CONTOUR_METHOD == 'sam':
                    sam_contours_x_batch, sam_contours_y_batch = generate_sam_contours_batch(
                        sam_mask_generator, images_pil, device
                    )
                elif CONTOUR_METHOD == 'fastsam':
                    sam_contours_x_batch, sam_contours_y_batch = generate_fastsam_contours_batch(
                        fastsam_model, images_pil, device
                    )
                else:  # slic
                    sam_contours_x_batch, sam_contours_y_batch = generate_slic_contours_batch(
                        images_pil, device
                    )

                sam_contours_x_batch = sam_contours_x_batch.to(device)
                sam_contours_y_batch = sam_contours_y_batch.to(device)
            
            # unary potential
            unary_loss = CollisionCrossEntropyLoss(segmentations, pseudolabel_probs)

            # pairwise potential
            pairwise_loss = PottsLoss(POTTS_TYPE, segmentations, sam_contours_x_batch, sam_contours_y_batch, use_color_diff=(CONTOUR_METHOD == 'color_diff'))

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
                    
                    val_output = model(val_transformed_image.unsqueeze(0))
                    segmentation = val_output['seg']
                    
                    update_miou(segmentation, val_target.unsqueeze(0), intersection_counts, union_counts, NUM_CLASSES, IGNORE_INDEX)

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
        model.load_state_dict(best_checkpoint['model_state_dict'], strict=False)
        print(f"Best model loaded successfully! Final validation mIoU: {best_miou:.4f}")
    
    vis_output_dir = os.path.join(DIRS['output'], DIRS['visualizations'])
    for i in range(0, len(original_train_dataset), config['visualization']['train_sample_interval']):
        vis_train_sample_img(
            original_train_dataset, train_dataset, model, i, vis_output_dir,
            text_emb_all=text_emb_all, num_all_fg=num_all_fg, num_bg=num_bg,
            fastsam_model=fastsam_model, sam_mask_generator=sam_mask_generator, num_classes=NUM_CLASSES,
            contour_method=CONTOUR_METHOD
        )
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