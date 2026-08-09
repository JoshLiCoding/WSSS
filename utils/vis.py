import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.loss import calculate_pairwise_affinity
from utils.dataset import cmap
from model.clip_pseudolabels import apply_hard_supervision
from utils.sam import (
    generate_sam_contours_batch,
    generate_fastsam_contours_batch,
    generate_slic_contours_batch,
    generate_gt_contours_batch,
)

def visualize_soft_probabilities(logits, softmax=True):
    if softmax:
        probabilities = logits.softmax(dim=0).detach().cpu().numpy()
    else:
        probabilities = logits.detach().cpu().numpy()
    num_classes, _, _ = probabilities.shape
    
    colors_array = cmap(normalized=True)  # Get normalized RGB values [0,1]
    colors_array = colors_array[:num_classes]  # Shape: [num_classes, 3]

    probabilities_expanded = np.expand_dims(probabilities, axis=-1)
    colors_reshaped = colors_array[:, np.newaxis, np.newaxis, :]
    weighted_colors = probabilities_expanded * colors_reshaped
    
    soft_vis = np.sum(weighted_colors, axis=0)
    soft_vis = np.clip(soft_vis*255, 0, 255).astype(np.uint8)
    return soft_vis

def visualize_hard_supervision_mask(hard_class_map, num_classes):
    """hard_class_map: [H, W] int array/tensor, -1 where the pixel kept its soft pseudolabel,
    else the class id it was hardened to. Returns an RGB image with a white background and
    each hardened pixel drawn in its class's solid color."""
    if isinstance(hard_class_map, torch.Tensor):
        hard_class_map = hard_class_map.cpu().numpy()
    colors_array = cmap()[:num_classes]
    vis = np.full((*hard_class_map.shape, 3), 255, dtype=np.uint8)
    hard_mask = hard_class_map >= 0
    vis[hard_mask] = colors_array[hard_class_map[hard_mask]]
    return vis

def vis_train_sample_img(original_train_dataset, train_dataset, model, index, output_dir,
                        cam_generator, text_emb_all, num_all_fg, num_bg, mask_generator, num_classes,
                        contour_method, hard_label_percentage):
    """
    Visualize training sample following the same procedure as the training loop.

    Args:
        original_train_dataset: Original dataset (for getting original image and GT)
        train_dataset: Transformed dataset (for getting transformed image and target)
        model: Segmentation model
        index: Index of sample to visualize
        output_dir: Directory to save visualization
        cam_generator: ClipGradCAM instance for pseudolabel generation
        text_emb_all: Precomputed text embeddings for all classes [num_all_classes, D]
        num_all_fg: Number of foreground classes
        num_bg: Number of background classes
        mask_generator: FastSAM or SAM mask generator instance, matching contour_method (can be None)
        num_classes: Number of classes
        hard_label_percentage: Fraction of pixels per image hardened to one-hot supervision
    """
    device = next(model.parameters()).device

    # Get original image and ground truth
    img, gt_mask = original_train_dataset[index]
    transformed_img, target = train_dataset[index]
    gt_mask = original_train_dataset.decode_target(gt_mask)

    model.eval()
    transformed_img_batch = transformed_img.unsqueeze(0).to(device)
    target_batch = [target]  # Wrap in list for batch processing

    # Generate pseudolabels via softmax-GradCAM (needs gradients enabled internally)
    pseudolabels_batch, class_indices_batch = cam_generator(
        transformed_img_batch, target_batch, text_emb_all, num_all_fg, num_bg
    )

    with torch.no_grad():
        # Forward pass through model to get segmentation
        model_outputs = model(transformed_img_batch)
        segmentations = model_outputs['seg']  # [1, C, H, W]

        # Convert pseudolabels to tensor format matching segmentation output
        _, _, H_seg, W_seg = segmentations.shape
        pseudolabel = pseudolabels_batch[0]
        class_indices = class_indices_batch[0]

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

        pseudolabel_probs_b, hard_class_map_b = apply_hard_supervision(
            pseudolabel_probs_b, hard_label_percentage
        )

        # Map to full class space
        pseudolabel_probs_vis = torch.zeros((num_classes, H_seg, W_seg), dtype=torch.float32, device=device)
        hard_class_map = torch.full((H_seg, W_seg), -1, dtype=torch.long, device=device)
        if len(class_indices) == 0:
            # Only background class
            pseudolabel_probs_vis[0] = pseudolabel_probs_b[0]  # background
            hard_class_map[hard_class_map_b == 0] = 0
        else:
            for idx, class_idx in enumerate(class_indices):
                pseudolabel_probs_vis[class_idx] = pseudolabel_probs_b[idx]
                hard_class_map[hard_class_map_b == idx] = int(class_idx)
            pseudolabel_probs_vis[0] = pseudolabel_probs_b[len(class_indices)]  # background
            hard_class_map[hard_class_map_b == len(class_indices)] = 0

        # Generate contours based on the selected method
        if contour_method == 'gt':
            target_batch_tensor = target.unsqueeze(0).to(device)  # [1, H, W]
            contours_x_batch, contours_y_batch = generate_gt_contours_batch(
                target_batch_tensor, device
            )
        elif contour_method in ('sam', 'fastsam', 'slic'):
            img_denorm = train_dataset.denormalize(transformed_img_batch[0].clone())
            img_np = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images_pil = [Image.fromarray(img_np)]

            if contour_method == 'sam':
                contours_x_batch, contours_y_batch = generate_sam_contours_batch(
                    mask_generator, images_pil, device
                )
            elif contour_method == 'fastsam':
                contours_x_batch, contours_y_batch = generate_fastsam_contours_batch(
                    mask_generator, images_pil, device
                )
            else:  # slic
                contours_x_batch, contours_y_batch = generate_slic_contours_batch(
                    images_pil, device
                )
        else:
            raise ValueError(
                f"Unknown contour_method: {contour_method}. Must be one of: 'sam', 'fastsam', 'slic', 'gt'"
            )
        
        contours_x = contours_x_batch[0].cpu().numpy()  # [H, W-1]
        contours_y = contours_y_batch[0].cpu().numpy()  # [H-1, W]
    
    # Prepare visualization data
    transformed_img_vis = train_dataset.denormalize(transformed_img_batch[0].cpu()).permute(1, 2, 0)
    segmentation_vis = segmentations[0]  # [C, H, W]
    
    # Visualize pseudolabel probabilities
    soft_pseudolabels = visualize_soft_probabilities(pseudolabel_probs_vis, softmax=False)
    pseudolabels_vis = pseudolabel_probs_vis.argmax(0).cpu().numpy().astype(np.uint8)
    pseudolabels_vis = Image.fromarray(pseudolabels_vis)
    pseudolabels_vis = original_train_dataset.decode_target(pseudolabels_vis)

    hard_supervision_vis = visualize_hard_supervision_mask(hard_class_map, num_classes)

    # Visualize segmentation output
    soft_output = visualize_soft_probabilities(segmentation_vis, softmax=True)
    output_vis = segmentation_vis.argmax(0).cpu().numpy().astype(np.uint8)
    output_vis = Image.fromarray(output_vis)
    output_vis = original_train_dataset.decode_target(output_vis)
    
    # Create visualization
    fig, axes = plt.subplots(7, 2, figsize=(8, 28))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    
    axes[0, 1].imshow(gt_mask)
    axes[0, 1].set_title('GT mask')

    axes[1, 0].imshow(transformed_img_vis)
    axes[1, 0].set_title('Transformed Image')

    axes[1, 1].imshow(hard_supervision_vis)
    axes[1, 1].set_title(f'Hard Supervision Pixels ({hard_label_percentage:.0%})')

    axes[2, 0].imshow(soft_pseudolabels)
    axes[2, 0].set_title('Soft Pseudolabels')

    axes[2, 1].imshow(pseudolabels_vis)
    axes[2, 1].set_title('Pseudolabels')

    contour_title = f'{contour_method.upper()} Contours'
    axes[3, 0].imshow(contours_x, cmap='gray', vmin=0, vmax=1)
    axes[3, 0].set_title(f'{contour_title} (horizontal)')

    axes[3, 1].imshow(contours_y, cmap='gray', vmin=0, vmax=1)
    axes[3, 1].set_title(f'{contour_title} (vertical)')

    contours_x_tensor = contours_x_batch[0:1].to(device)  # [1, H, W-1]
    contours_y_tensor = contours_y_batch[0:1].to(device)  # [1, H-1, W]
    w_x_vis = calculate_pairwise_affinity(contours_x_tensor).squeeze(0).cpu().numpy()
    w_y_vis = calculate_pairwise_affinity(contours_y_tensor).squeeze(0).cpu().numpy()
    axes[4, 0].imshow(w_x_vis, cmap='gray', vmin=0, vmax=1)
    axes[4, 0].set_title(f'{contour_title} Distance Field (horizontal)')

    axes[4, 1].imshow(w_y_vis, cmap='gray', vmin=0, vmax=1)
    axes[4, 1].set_title(f'{contour_title} Distance Field (vertical)')

    axes[5, 0].imshow(soft_output)
    axes[5, 0].set_title('Soft Model Output')

    axes[5, 1].imshow(output_vis)
    axes[5, 1].set_title('Hard Model Output')
    
    expanded_contours_x = np.zeros((H_seg, W_seg), dtype=np.float32)
    expanded_contours_x[:, :contours_x.shape[1]] = contours_x
    axes[6, 0].imshow(expanded_contours_x, alpha=0.5, cmap='gray')
    axes[6, 0].imshow(soft_output, alpha=0.5)
    axes[6, 0].set_title(f'Soft Model Output & {contour_title} (horizontal)')

    expanded_contours_y = np.zeros((H_seg, W_seg), dtype=np.float32)
    expanded_contours_y[:contours_y.shape[0], :] = contours_y
    axes[6, 1].imshow(expanded_contours_y, alpha=0.5, cmap='gray')
    axes[6, 1].imshow(soft_output, alpha=0.5)
    axes[6, 1].set_title(f'Soft Model Output & {contour_title} (vertical)')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'visualization_sample_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training visualization saved as '{save_path}'")

def vis_val_sample_img(original_val_dataset, val_dataset, model, index, output_dir='.'):
    device = next(model.parameters()).device
    
    img, gt_mask = original_val_dataset[index]
    transformed_img, _ = val_dataset[index]

    gt_mask = original_val_dataset.decode_target(gt_mask)
    
    model.eval()
    transformed_img = transformed_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(transformed_img)['seg'][0].cpu()
    
    soft_output = visualize_soft_probabilities(output, softmax=True)

    output_vis = output.argmax(0).numpy().astype(np.uint8)
    output_vis = Image.fromarray(output_vis)
    output_vis = original_val_dataset.decode_target(output_vis)
    
    # Resize model output to original image size
    output_resized = output.unsqueeze(0)
    output_resized = torch.nn.functional.interpolate(
        output_resized, size=(img.size[1], img.size[0]), mode='bilinear', align_corners=False
    )[0]
    output_resized_vis = output_resized.argmax(0).numpy().astype(np.uint8)
    output_resized_vis = Image.fromarray(output_resized_vis)
    output_resized_vis = original_val_dataset.decode_target(output_resized_vis)
    soft_output_resized = visualize_soft_probabilities(output_resized, softmax=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Top row: original image size
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')

    axes[0, 1].imshow(gt_mask)
    axes[0, 1].set_title('GT mask')

    axes[0, 2].imshow(output_resized_vis)
    axes[0, 2].set_title('Model Output (Resized)')

    axes[0, 3].imshow(soft_output_resized)
    axes[0, 3].set_title('Soft Model Output (Resized)')

    # Bottom row: native segmentation output size
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(output_vis)
    axes[1, 2].set_title('Model Output')

    axes[1, 3].imshow(soft_output)
    axes[1, 3].set_title('Soft Model Output')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'val_visualization_sample_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Validation visualization saved as '{save_path}'")

def vis_train_loss(num_epochs, epoch_total_losses, epoch_unary_losses, epoch_pairwise_losses, output_dir='.'):
    # Graph 1: Total Loss
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, epoch_total_losses, label='Total Loss', color='blue', linewidth=2)
    plt.title('Total Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'train_total_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Graph 2: Individual Loss Components
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, epoch_unary_losses, label='Unary Loss', color='green', linestyle='--')
    plt.plot(epochs, epoch_pairwise_losses, label='Pairwise Loss', color='red', linestyle='--')

    plt.title('Individual Loss Components Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'train_individual_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss visualizations saved")

def vis_val_loss(validation_mious, validation_epochs, output_dir='.'):
    plt.figure(figsize=(6, 4))
    plt.plot(validation_epochs, validation_mious, label='Validation mIoU', color='purple', marker='o')
    plt.title('Validation mIoU Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'val_miou.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Validation mIoU visualization saved")