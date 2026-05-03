import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.loss import calculate_pairwise_affinity
from utils.dataset import cmap
from model.dino_txt_full_img import generate_pseudolabels_batch
from utils.sam import (
    generate_sam_contours_batch,
    generate_fastsam_contours_batch,
    generate_slic_contours_batch,
    generate_gt_contours_batch,
    generate_color_diff_contours_batch,
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

def vis_train_sample_img(original_train_dataset, train_dataset, model, index, output_dir, 
                        text_emb_all, num_all_fg, num_bg, fastsam_model, sam_mask_generator, num_classes,
                        contour_method):
    """
    Visualize training sample following the same procedure as the training loop.
    
    Args:
        original_train_dataset: Original dataset (for getting original image and GT)
        train_dataset: Transformed dataset (for getting transformed image and target)
        model: Segmentation model
        index: Index of sample to visualize
        output_dir: Directory to save visualization
        text_emb_all: Precomputed text embeddings for all classes [num_all_classes, D]
        num_all_fg: Number of foreground classes
        num_bg: Number of background classes
        fastsam_model: FastSAM model instance
        sam_mask_generator: SAM automatic mask generator instance (can be None)
        num_classes: Number of classes
    """
    device = next(model.parameters()).device
    
    # Get original image and ground truth
    img, gt_mask = original_train_dataset[index]
    transformed_img, target = train_dataset[index]
    gt_mask = original_train_dataset.decode_target(gt_mask)

    model.eval()
    transformed_img_batch = transformed_img.unsqueeze(0).to(device)
    target_batch = [target]  # Wrap in list for batch processing
    
    with torch.no_grad():
        model_outputs = model(transformed_img_batch)
        segmentations = model_outputs["seg"]
        clip_patch_tokens = model_outputs["clip"]

        pseudolabels_batch, class_indices_batch = generate_pseudolabels_batch(
            clip_patch_tokens, target_batch, text_emb_all, num_all_fg, num_bg
        )
        
        # Convert pseudolabels to tensor format matching segmentation output
        B, _, H_seg, W_seg = segmentations.shape
        pseudolabel_probs = torch.zeros((B, num_classes, H_seg, W_seg), dtype=torch.float32, device=device)
        
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
            
            # Map to full class space [num_classes, H_seg, W_seg]
            if len(class_indices) == 0:
                # Only background class
                pseudolabel_probs[b, 0] = pseudolabel_probs_b[0]  # background
            else:
                for idx, class_idx in enumerate(class_indices):
                    pseudolabel_probs[b, class_idx] = pseudolabel_probs_b[idx]
                pseudolabel_probs[b, 0] = pseudolabel_probs_b[len(class_indices)]  # background
        
        pseudolabel_probs_vis = pseudolabel_probs[0]  # [num_classes, H_seg, W_seg]
        
        # Generate contours based on the selected method
        if contour_method == 'gt':
            target_batch_tensor = target.unsqueeze(0).to(device)  # [1, H, W]
            sam_contours_x_batch, sam_contours_y_batch = generate_gt_contours_batch(
                target_batch_tensor, device
            )
        elif contour_method == 'color_diff':
            img_denorm = train_dataset.denormalize(transformed_img_batch[0].clone())
            H_seg, W_seg = segmentations.shape[2], segmentations.shape[3]
            img_denorm_batch = img_denorm.unsqueeze(0)  # [1, C, H_orig, W_orig]
            img_denorm_batch = F.interpolate(img_denorm_batch, size=(H_seg, W_seg), mode='bilinear', align_corners=False).to(device)
            sam_contours_x_batch, sam_contours_y_batch = generate_color_diff_contours_batch(img_denorm_batch, device)
        elif contour_method in ('sam', 'fastsam', 'slic'):
            img_denorm = train_dataset.denormalize(transformed_img_batch[0].clone())
            img_np = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images_pil = [Image.fromarray(img_np)]

            if contour_method == 'sam':
                sam_contours_x_batch, sam_contours_y_batch = generate_sam_contours_batch(
                    sam_mask_generator, images_pil, device
                )
            elif contour_method == 'fastsam':
                sam_contours_x_batch, sam_contours_y_batch = generate_fastsam_contours_batch(
                    fastsam_model, images_pil, device
                )
            else:  # slic
                sam_contours_x_batch, sam_contours_y_batch = generate_slic_contours_batch(
                    images_pil, device
                )
        else:
            raise ValueError(
                f"Unknown contour_method: {contour_method}. Must be one of: 'sam', 'fastsam', 'slic', 'gt', 'color_diff'"
            )
        
        sam_contours_x = sam_contours_x_batch[0].cpu().numpy()  # [H, W-1]
        sam_contours_y = sam_contours_y_batch[0].cpu().numpy()  # [H-1, W]
    
    # Prepare visualization data
    transformed_img_vis = train_dataset.denormalize(transformed_img_batch[0].cpu()).permute(1, 2, 0)
    segmentation_vis = segmentations[0]  # [C, H, W]
    
    # Visualize pseudolabel probabilities
    soft_pseudolabels = visualize_soft_probabilities(pseudolabel_probs_vis, softmax=False)
    pseudolabels_vis = pseudolabel_probs_vis.argmax(0).cpu().numpy().astype(np.uint8)
    pseudolabels_vis = Image.fromarray(pseudolabels_vis)
    pseudolabels_vis = original_train_dataset.decode_target(pseudolabels_vis)
    
    # Visualize segmentation output
    soft_output = visualize_soft_probabilities(segmentation_vis, softmax=True)
    output_vis = segmentation_vis.argmax(0).cpu().numpy().astype(np.uint8)
    output_vis = Image.fromarray(output_vis)
    output_vis = original_train_dataset.decode_target(output_vis)
    
    # Create visualization
    fig, axes = plt.subplots(8, 2, figsize=(8, 32))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    
    axes[0, 1].imshow(gt_mask)
    axes[0, 1].set_title('GT mask')

    axes[1, 0].imshow(transformed_img_vis)
    axes[1, 0].set_title('Transformed Image')

    axes[1, 1].axis('off')

    axes[2, 0].imshow(soft_pseudolabels)
    axes[2, 0].set_title('Soft Pseudolabels')

    axes[2, 1].imshow(pseudolabels_vis)
    axes[2, 1].set_title('Pseudolabels')

    contour_title = f'{contour_method.upper()} Contours'
    axes[3, 0].imshow(sam_contours_x, cmap='gray', vmin=0, vmax=1)
    axes[3, 0].set_title(f'{contour_title} (horizontal)')

    axes[3, 1].imshow(sam_contours_y, cmap='gray', vmin=0, vmax=1)
    axes[3, 1].set_title(f'{contour_title} (vertical)')

    sam_contours_x_tensor = sam_contours_x_batch[0:1].to(device)  # [1, H, W-1]
    sam_contours_y_tensor = sam_contours_y_batch[0:1].to(device)  # [1, H-1, W]
    # For color_diff, contours are already weights; for others, negate to get distance field
    w_x_vis = sam_contours_x_tensor.squeeze(0).cpu().numpy() if contour_method == 'color_diff' else calculate_pairwise_affinity(sam_contours_x_tensor).squeeze(0).cpu().numpy()
    w_y_vis = sam_contours_y_tensor.squeeze(0).cpu().numpy() if contour_method == 'color_diff' else calculate_pairwise_affinity(sam_contours_y_tensor).squeeze(0).cpu().numpy()
    axes[4, 0].imshow(w_x_vis, cmap='gray', vmin=0, vmax=1)
    axes[4, 0].set_title(f'{contour_title} Distance Field (horizontal)')

    axes[4, 1].imshow(w_y_vis, cmap='gray', vmin=0, vmax=1)
    axes[4, 1].set_title(f'{contour_title} Distance Field (vertical)')

    axes[5, 0].imshow(soft_output)
    axes[5, 0].set_title('Soft Model Output')

    axes[5, 1].imshow(output_vis)
    axes[5, 1].set_title('Hard Model Output')
    
    # Get output dimensions from segmentation (H_seg, W_seg)
    H, W = segmentation_vis.shape[1], segmentation_vis.shape[2]
    # SAM contours are already at segmentation size (4x downsampled)
    # Expand SAM contours to full spatial dimensions for overlay
    expanded_sam_contours_x = np.zeros((H, W), dtype=np.float32)
    expanded_sam_contours_x[:, :sam_contours_x.shape[1]] = sam_contours_x
    axes[6, 0].imshow(expanded_sam_contours_x, alpha=0.5, cmap='gray')
    axes[6, 0].imshow(soft_output, alpha=0.5)
    axes[6, 0].set_title(f'Soft Model Output & {contour_title} (horizontal)')

    expanded_sam_contours_y = np.zeros((H, W), dtype=np.float32)
    expanded_sam_contours_y[:sam_contours_y.shape[0], :] = sam_contours_y
    axes[6, 1].imshow(expanded_sam_contours_y, alpha=0.5, cmap='gray')
    axes[6, 1].imshow(soft_output, alpha=0.5)
    axes[6, 1].set_title(f'Soft Model Output & {contour_title} (vertical)')

    inverted_contours_or = np.minimum(1.0 - expanded_sam_contours_x, 1.0 - expanded_sam_contours_y)
    axes[7, 0].imshow(inverted_contours_or, cmap='gray', vmin=0, vmax=1)
    axes[7, 0].set_title(f'Inverted {contour_title} (H|V OR)')

    expanded_w_x = np.zeros((H, W), dtype=np.float32)
    expanded_w_x[:, :w_x_vis.shape[1]] = w_x_vis
    expanded_w_y = np.zeros((H, W), dtype=np.float32)
    expanded_w_y[:w_y_vis.shape[0], :] = w_y_vis
    distance_field_or = np.minimum(expanded_w_x, expanded_w_y)
    axes[7, 1].imshow(distance_field_or, cmap='gray', vmin=0, vmax=1)
    axes[7, 1].set_title(f'{contour_title} Distance Field (H|V OR)')

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
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')

    axes[0, 2].imshow(gt_mask)
    axes[0, 2].set_title('GT mask')

    axes[1, 0].imshow(output_vis)
    axes[1, 0].set_title('Model Output')

    axes[1, 1].imshow(soft_output)
    axes[1, 1].set_title('Soft Model Output')

    axes[1, 2].imshow(output_resized_vis)
    axes[1, 2].set_title('Model Output (Resized)')

    axes[1, 3].imshow(soft_output_resized)
    axes[1, 3].set_title('Soft Model Output (Resized)')

    axes[0, 1].axis('off')
    axes[0, 3].axis('off')

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