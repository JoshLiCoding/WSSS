"""Stage 2: train ResNet101 DeepLabV2 on cached dino.txt pseudo-labels.

Stage 1 (dino.txt pseudo-labels + mask boundaries) runs here too if its caches are missing;
run generate.py to do it on its own beforehand.
"""

import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.deeplabv2 import DeepLabV2
from model.dino_txt import ensure_pseudolabels
from model.scheduler import PolyLR
from utils.boundaries import ensure_boundaries
from utils.dataset import CustomSegmentationTrain, CustomSegmentationVal, build_dataset
from utils.loss import CollisionCrossEntropyLoss, KLDivergenceLoss, PottsLoss
from utils.metrics import update_miou
from utils.pseudolabels import soft_pseudolabels
from utils.vis import vis_train_loss, vis_train_sample_img, vis_val_loss, vis_val_sample_img

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

    original_train_dataset = build_dataset(cfg, 'train')  # augmented VOC train set
    original_val_dataset = build_dataset(cfg, 'val')

    # Stage 1: generate the pseudo-labels and boundaries that are not cached yet.
    ensure_pseudolabels(cfg, original_train_dataset, cfg.pseudolabel.dir, device)
    ensure_boundaries(cfg, original_train_dataset, cfg.boundary.dir, device)

    NUM_CLASSES = cfg.model.num_classes
    LABEL_SIZE = cfg.dataset.label_size
    train_dataset = CustomSegmentationTrain(
        original_train_dataset,
        resize_size=cfg.dataset.resize_size,
        label_size=LABEL_SIZE,
        num_classes=NUM_CLASSES,
        pseudolabel_dir=cfg.pseudolabel.dir,
        boundary_dir=cfg.boundary.dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_dataset = CustomSegmentationVal(original_val_dataset, resize_size=cfg.dataset.resize_size)

    model = DeepLabV2(
        num_classes=NUM_CLASSES,
        pretrained_backbone=cfg.model.pretrained_backbone,
    ).to(device)

    LEARNING_RATE = cfg.training.learning_rate
    optimizer = torch.optim.SGD(
        params=model.param_groups(LEARNING_RATE, cfg.training.weight_decay),
        momentum=cfg.training.momentum,
    )
    NUM_EPOCHS = cfg.training.num_epochs
    max_iters = max(1, NUM_EPOCHS * len(train_loader))
    scheduler = PolyLR(optimizer, max_iters=max_iters) if cfg.training.poly_lr else None

    model_checkpoint = cfg.paths.model_checkpoint
    if os.path.exists(model_checkpoint):
        print(f"Loading checkpoint from {model_checkpoint}...")
        checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
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

    TRAIN_ONLY = cfg.training.train_only
    MODEL_PATH = cfg.paths.model
    TEMPERATURE = cfg.pseudolabel.temperature
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training epochs"):
        model.train()

        running_total_loss = 0.0
        running_unary_loss = 0.0
        running_pairwise_loss = 0.0
        for i, (images, targets, pseudolabel_sims, present, edges) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()

            segmentations = model(images)
            if segmentations.shape[-2:] != (LABEL_SIZE, LABEL_SIZE):
                segmentations = F.interpolate(
                    segmentations, size=(LABEL_SIZE, LABEL_SIZE), mode='bilinear', align_corners=False
                )

            # Soft pseudo-labels from the cached dino.txt similarities
            pseudolabel_probs = soft_pseudolabels(
                pseudolabel_sims.to(device), present.to(device), TEMPERATURE
            )

            # Cached mask boundaries, split into the horizontal and vertical neighbour pairs
            edges = edges.to(device).float()
            contours_x_batch = edges[:, 0, :, :-1]  # (B, H, W-1)
            contours_y_batch = edges[:, 1, :-1, :]  # (B, H-1, W)

            # unary potential
            unary_loss = CollisionCrossEntropyLoss(segmentations, pseudolabel_probs) # KLDivergenceLoss(segmentations, pseudolabel_probs) 

            # pairwise potential
            pairwise_loss = PottsLoss(segmentations, contours_x_batch, contours_y_batch)

            total_loss = unary_loss + pairwise_loss

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
                torch.save({'model_state_dict': model.state_dict()}, MODEL_PATH)
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

                    segmentation = model(val_transformed_image.unsqueeze(0))

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
                torch.save({'model_state_dict': model.state_dict()}, MODEL_PATH)
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
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Best model loaded successfully! Final validation mIoU: {best_miou:.4f}")

    vis_output_dir = os.path.join(DIRS['output'], DIRS['visualizations'])
    for i in range(0, len(original_train_dataset), cfg.visualization.train_sample_interval):
        vis_train_sample_img(
            original_train_dataset, train_dataset, model, i, vis_output_dir,
            num_classes=NUM_CLASSES, temperature=TEMPERATURE,
            contour_method=cfg.loss.contour_method,
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
