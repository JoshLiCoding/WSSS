import os
import sys

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import FastSAM

from model.dino import DinoWSSS
from model.dino_txt_full_img import build_text_embeddings, generate_pseudolabels_batch, get_class_names
from utils.dataset import CustomSegmentationTrain, CustomSegmentationVal, build_dataset
from utils.loss import PottsLoss, get_unary_loss, load_class_weights
from utils.metrics import update_miou
from utils.sam import generate_contours_batch
from utils.vis import vis_train_sample_img, vis_val_sample_img, vis_train_loss, vis_val_loss

DINOV3_LOCATION = '/u501/j234li/reg_loss/model/dinov3'
sys.path.append(DINOV3_LOCATION)

VOC_CLASS_NAMES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}

COCO_CLASS_NAMES = {0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat", 36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle", 41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 80: "toothbrush", 255: "ignore"}


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASS_NAMES = VOC_CLASS_NAMES if cfg.dataset.dataset_name == 'voc' else COCO_CLASS_NAMES

    # Setup directories (interpolations are resolved by OmegaConf)
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

    NUM_CLASSES = cfg.model.num_classes
    train_dataset = CustomSegmentationTrain(
        original_train_dataset, num_classes=NUM_CLASSES, resize_size=cfg.dataset.resize_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_dataset = CustomSegmentationVal(original_val_dataset, resize_size=cfg.dataset.resize_size)

    CONTOUR_METHOD = cfg.loss.contour_method
    DILATION_SIZE = cfg.loss.dilation_size
    fastsam_model = FastSAM('FastSAM-x.pt') if CONTOUR_METHOD == 'fastsam' else None
    sam_mask_generator = None
    if CONTOUR_METHOD == 'sam':
        sam_model = sam_model_registry["vit_b"](checkpoint=cfg.paths.sam_checkpoint).to(device)
        sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=16,
            points_per_batch=256,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8
        )
        print("SAM model initialized for automatic mask generation")

    # Initialize DinoTxt model and tokenizer for pseudolabel generation
    text_model, tokenizer = torch.hub.load(
        DINOV3_LOCATION,
        'dinov3_vitl16_dinotxt_tet1280d20h24l',
        source='local',
        weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'),
        backbone_weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    )
    text_model = text_model.to(device)
    text_model.eval()

    # Precompute text embeddings for all classes (once before training)
    print("Precomputing text embeddings for all classes...")
    fg_class_names, bg_class_names = get_class_names(cfg.dataset.dataset_name)
    text_emb_all = build_text_embeddings(text_model, tokenizer, fg_class_names + bg_class_names, device)
    NUM_FG = len(fg_class_names)
    print(f"Text embeddings computed: shape {text_emb_all.shape}")

    model = DinoWSSS(
        backbone_name=cfg.model.backbone_name,
        num_transformer_blocks=cfg.model.num_transformer_blocks,
        num_conv_blocks=cfg.model.num_conv_blocks,
        out_channels=cfg.model.out_channels,
        use_bottleneck=cfg.model.use_bottleneck,
        use_transpose_conv=cfg.model.use_transpose_conv
    ).to(device)
    model.backbone.eval()
    model.dinotxt_head.eval()

    LEARNING_RATE = cfg.training.learning_rate
    decoder = [model.transformer_blocks, model.ln, model.conv_blocks, model.lin_classifier]
    if cfg.model.use_transpose_conv:
        decoder += [model.upsample_conv1, model.upsample_conv2]

    optimizer = torch.optim.SGD(
        params=[{'params': module.parameters(), 'lr': LEARNING_RATE} for module in decoder],
        lr=LEARNING_RATE,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay
    )

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

    NUM_EPOCHS = cfg.training.num_epochs
    TRAIN_ONLY = cfg.training.train_only
    MODEL_PATH = cfg.paths.model
    TEMPERATURE = cfg.pseudolabel.temperature
    MIN_MAX = cfg.pseudolabel.min_max
    class_weights = load_class_weights(cfg, NUM_CLASSES).to(device)
    unary_loss_fn = get_unary_loss(cfg.loss.unary_method)

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training epochs"):
        model.train()
        model.backbone.eval()
        model.dinotxt_head.eval()

        running_total_loss = 0.0
        running_unary_loss = 0.0
        running_pairwise_loss = 0.0
        for i, (images, targets, present) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass through model to get segmentation and dino.txt patch tokens
            model_outputs = model(images)
            segmentations = model_outputs['seg']
            size = segmentations.shape[-2:]

            # Pseudolabels from the dino.txt patch tokens, over the image-level annotation tags
            pseudolabel_probs = generate_pseudolabels_batch(
                model_outputs['dinotxt'], present.to(device), text_emb_all, NUM_FG,
                size, TEMPERATURE, MIN_MAX
            )

            contours_x, contours_y = generate_contours_batch(
                CONTOUR_METHOD, images, targets, train_dataset.denormalize, size, device,
                sam_mask_generator=sam_mask_generator, fastsam_model=fastsam_model
            )

            # unary potential
            unary_loss = unary_loss_fn(segmentations, pseudolabel_probs)

            # pairwise potential
            pairwise_loss = PottsLoss(segmentations, contours_x, contours_y,
                                      class_weights, DILATION_SIZE,
                                      use_color_diff=(CONTOUR_METHOD == 'color_diff'))

            total_loss = unary_loss + pairwise_loss

            total_loss.backward()
            optimizer.step()

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

                    segmentation = model(val_transformed_image.unsqueeze(0))['seg']

                    update_miou(segmentation, val_target.unsqueeze(0), intersection_counts, union_counts, NUM_CLASSES, IGNORE_INDEX)

            ious = []
            for cls in range(NUM_CLASSES):
                if union_counts[cls] == 0:
                    continue
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
        model.load_state_dict(best_checkpoint['model_state_dict'], strict=False)
        print(f"Best model loaded successfully! Final validation mIoU: {best_miou:.4f}")

    vis_output_dir = os.path.join(DIRS['output'], DIRS['visualizations'])
    for i in range(0, len(original_train_dataset), cfg.visualization.train_sample_interval):
        vis_train_sample_img(
            original_train_dataset, train_dataset, model, i, vis_output_dir,
            text_emb_all=text_emb_all, num_fg=NUM_FG, temperature=TEMPERATURE, min_max=MIN_MAX,
            fastsam_model=fastsam_model, sam_mask_generator=sam_mask_generator,
            contour_method=CONTOUR_METHOD, dilation_size=DILATION_SIZE
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
