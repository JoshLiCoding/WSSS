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
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from ultralytics import FastSAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from model.clip_wsss import ClipWSSS
from model.deeplab import deeplabv3_resnet101, deeplabv3plus_resnet101
from model.scheduler import PolyLR
from model.clip_pseudolabels import ClipGradCAM, build_text_embeddings, apply_hard_supervision
from utils.dataset import VOCSegmentation, COCOSegmentation, CustomSegmentationTrain, CustomSegmentationVal
from utils.loss import CollisionCrossEntropyLoss, PottsLoss, KLDivergenceLoss, CrossEntropyLoss
from utils.metrics import update_miou
from utils.sam import (
    generate_sam_contours_batch,
    generate_fastsam_contours_batch,
    generate_slic_contours_batch,
    generate_gt_contours_batch,
)
from utils.vis import vis_train_sample_img, vis_val_sample_img, vis_train_loss, vis_val_loss

def trainable_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('clip.')}


VOC_CLASS_NAMES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}

COCO_CLASS_NAMES = {0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat", 36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle", 41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 80: "toothbrush", 255: "ignore"}


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASS_NAMES = VOC_CLASS_NAMES if cfg.dataset.dataset_name == 'voc' else COCO_CLASS_NAMES

    # Setup directories and paths (num_epochs interpolation is resolved by OmegaConf)
    DIRS = {
        'output': cfg.directories.output,
        'checkpoints': cfg.directories.checkpoints,
        'visualizations': cfg.directories.visualizations,
    }
    for dir_name, dir_path in DIRS.items():
        full_path = os.path.join(DIRS['output'], dir_path) if dir_name != 'output' else dir_path
        os.makedirs(full_path, exist_ok=True)

    # Print configuration parameters
    print(OmegaConf.to_yaml(cfg))

    # Initialize Weights & Biases
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=DIRS['visualizations'],
        config=OmegaConf.to_container(cfg, resolve=True),  # Log the entire configuration
    )

    # augmented VOC train set
    if cfg.dataset.dataset_name == 'voc':
        original_train_dataset = VOCSegmentation(
            cfg.dataset.root,
            image_set='train',
        )
    elif cfg.dataset.dataset_name == 'coco':
        original_train_dataset = COCOSegmentation(
            cfg.dataset.root,
            image_set='train',
        )

    if cfg.dataset.dataset_name == 'voc':
        original_val_dataset = VOCSegmentation(
            cfg.dataset.root,
            image_set='val',
        )
    elif cfg.dataset.dataset_name == 'coco':
        original_val_dataset = COCOSegmentation(
            cfg.dataset.root,
            image_set='val',
        )

    train_dataset = CustomSegmentationTrain(
        original_train_dataset,
        resize_size=cfg.dataset.resize_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_dataset = CustomSegmentationVal(original_val_dataset, resize_size=cfg.dataset.resize_size)

    # Initialize the mask generator for the selected contour method (if needed)
    CONTOUR_METHOD = cfg.loss.contour_method
    mask_generator = None
    if CONTOUR_METHOD == 'fastsam':
        mask_generator = FastSAM('FastSAM-x.pt')
    elif CONTOUR_METHOD == 'sam':
        sam_checkpoint = 'sam_checkpoint/sam_vit_b_01ec64.pth'
        model_type = "vit_b"  # or "vit_b", "vit_l" depending on checkpoint
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=16,
            points_per_batch=256,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8
        )
        print("SAM model initialized for automatic mask generation")
    
    NUM_CLASSES = cfg.model.num_classes
    model = ClipWSSS(
        clip_name=cfg.model.clip_name,
        num_transformer_blocks=cfg.model.num_transformer_blocks,
        num_conv_blocks=cfg.model.num_conv_blocks,
        out_channels=NUM_CLASSES,
        img_size=cfg.dataset.resize_size,
        device=device,
    ).to(device)
    cam_generator = ClipGradCAM(model.clip, img_size=cfg.dataset.resize_size, bg_exponent=cfg.pseudolabel.bg_exponent)

    # Precompute text embeddings for all classes (once before training)
    print("Precomputing text embeddings for all classes...")
    text_emb_all, num_all_fg, num_bg = build_text_embeddings(cfg, model.clip, device=device)
    print(f"Text embeddings computed: shape {text_emb_all.shape}")

    LEARNING_RATE = cfg.training.learning_rate
    optimizer_params = [
        {'params': model.transformer_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.ln.parameters(), 'lr': LEARNING_RATE},
        {'params': model.conv_blocks.parameters(), 'lr': LEARNING_RATE},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE},
    ]

    optimizer = torch.optim.SGD(
        params=optimizer_params,
        lr=LEARNING_RATE,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay
    )
    NUM_EPOCHS = cfg.training.num_epochs
    max_iters = NUM_EPOCHS * len(train_loader)
    # scheduler = PolyLR(optimizer, max_iters=max_iters)

    model_checkpoint = cfg.paths.model_checkpoint
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

    TRAIN_ONLY = cfg.training.train_only
    MODEL_PATH = cfg.paths.model
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training epochs"):
        model.train()
        model.clip.eval()

        running_total_loss = 0.0
        running_unary_loss = 0.0
        running_pairwise_loss = 0.0
        for i, (transformed_images, targets) in enumerate(train_loader):
            transformed_images = transformed_images.to(device)

            optimizer.zero_grad()

            # Forward pass through model (frozen CLIP backbone + trainable decoder)
            model_outputs = model(transformed_images)
            segmentations = model_outputs['seg']

            # Generate pseudolabels via softmax-GradCAM on the frozen CLIP backbone
            pseudolabels_batch, class_indices_batch = cam_generator(
                transformed_images, targets, text_emb_all, num_all_fg, num_bg
            )
            
            # Convert pseudolabels to tensor format matching segmentation output
            B, _, H_seg, W_seg = segmentations.shape
            pseudolabel_probs = torch.zeros((B, NUM_CLASSES, H_seg, W_seg), dtype=torch.float32, device=device)

            for b in range(B):
                pseudolabel = pseudolabels_batch[b]
                class_indices = class_indices_batch[b]

                pseudolabel_tensor = pseudolabel.unsqueeze(0)

                # Interpolate to segmentation size
                pseudolabel_tensor = F.interpolate(pseudolabel_tensor, size=(H_seg, W_seg), mode='bilinear', align_corners=False)
                pseudolabel_tensor = pseudolabel_tensor[0]
                
                # Softmax with temperature
                t = 1.0
                pseudolabel_probs_b = torch.softmax(pseudolabel_tensor / t, dim=0)

                # Min-max normalize channel-wise, then renormalize to probability simplex
                min_vals = pseudolabel_probs_b.view(pseudolabel_probs_b.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
                max_vals = pseudolabel_probs_b.view(pseudolabel_probs_b.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
                pseudolabel_probs_b = (pseudolabel_probs_b - min_vals) / (max_vals - min_vals + 1e-8)

                pseudolabel_probs_b = pseudolabel_probs_b / (pseudolabel_probs_b.sum(dim=0, keepdim=True) + 1e-8)

                pseudolabel_probs_b, _ = apply_hard_supervision(
                    pseudolabel_probs_b, cfg.pseudolabel.hard_label_percentage
                )

                # Map to full class space
                if len(class_indices) == 0:
                    # Only background class
                    pseudolabel_probs[b, 0] = pseudolabel_probs_b[0]  # background
                else:
                    for idx, class_idx in enumerate(class_indices):
                        pseudolabel_probs[b, class_idx] = pseudolabel_probs_b[idx]
                    pseudolabel_probs[b, 0] = pseudolabel_probs_b[len(class_indices)]  # background
            
            # Generate contours based on the selected method
            if CONTOUR_METHOD == 'gt':
                contours_x_batch, contours_y_batch = generate_gt_contours_batch(targets, device)
            elif CONTOUR_METHOD in ('sam', 'fastsam', 'slic'):
                def _to_pil(img_t):
                    img_denorm = train_dataset.denormalize(img_t.clone())
                    img_np = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    return Image.fromarray(img_np)

                images_pil = [_to_pil(img) for img in transformed_images]

                if CONTOUR_METHOD == 'sam':
                    contours_x_batch, contours_y_batch = generate_sam_contours_batch(
                        mask_generator, images_pil, device
                    )
                elif CONTOUR_METHOD == 'fastsam':
                    contours_x_batch, contours_y_batch = generate_fastsam_contours_batch(
                        mask_generator, images_pil, device
                    )
                else:
                    contours_x_batch, contours_y_batch = generate_slic_contours_batch(
                        images_pil, device
                    )

                contours_x_batch = contours_x_batch.to(device)
                contours_y_batch = contours_y_batch.to(device)
            
            # unary potential
            unary_loss = CollisionCrossEntropyLoss(segmentations, pseudolabel_probs) # KLDivergenceLoss(segmentations, pseudolabel_probs)

            # pairwise potential
            pairwise_loss = PottsLoss(segmentations, contours_x_batch, contours_y_batch) # torch.tensor(0.0, device=device)

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
        if (epoch + 1) % cfg.training.validation_interval == 0 or epoch == NUM_EPOCHS - 1:
            if TRAIN_ONLY:
                print("TRAIN_ONLY is set to True, skipping validation. Saving model checkpoint...")
                torch.save({
                    'model_state_dict': trainable_state_dict(model)
                }, MODEL_PATH)
                continue
            
            print(f"Running validation at epoch {epoch + 1}...")
            model.eval()

            IGNORE_INDEX = cfg.training.ignore_index
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
                    'model_state_dict': trainable_state_dict(model)
                }, MODEL_PATH)
                print(f"New best model saved! mIoU: {best_miou:.4f} at epoch {best_epoch}")
                

    print(f"\nTraining complete! Best model was at epoch {best_epoch} with mIoU {best_miou:.4f}")
    
    # Log final summary to wandb
    wandb.log({
        "final/best_miou": best_miou,
        "final/best_epoch": best_epoch,
        "final/total_epochs": NUM_EPOCHS
    })
    
    if os.path.exists(MODEL_PATH):
        best_checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'], strict=False)
        print(f"Best model loaded successfully! Final validation mIoU: {best_miou:.4f}")
    
    vis_output_dir = os.path.join(DIRS['output'], DIRS['visualizations'])
    for i in range(0, len(original_train_dataset), cfg.visualization.train_sample_interval):
        vis_train_sample_img(
            original_train_dataset, train_dataset, model, i, vis_output_dir,
            cam_generator=cam_generator, text_emb_all=text_emb_all, num_all_fg=num_all_fg, num_bg=num_bg,
            mask_generator=mask_generator, num_classes=NUM_CLASSES,
            contour_method=CONTOUR_METHOD, hard_label_percentage=cfg.pseudolabel.hard_label_percentage
        )
    vis_train_loss(NUM_EPOCHS, epoch_total_losses, epoch_unary_losses, epoch_pairwise_losses, vis_output_dir)
    
    if not TRAIN_ONLY:
        for i in range(0, len(original_val_dataset), cfg.visualization.val_sample_interval):
            vis_val_sample_img(original_val_dataset, val_dataset, model, i, vis_output_dir)
        vis_val_loss(validation_mious, validation_epochs, vis_output_dir)

    # Log visualizations to wandb
    if cfg.wandb.log_visualizations:
        for file in os.listdir(vis_output_dir):
            if file.endswith('.png'):
                wandb.log({"plots/visualizations": wandb.Image(os.path.join(vis_output_dir, file))})
        
    wandb.finish()

if __name__ == "__main__":
    main()