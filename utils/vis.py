import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.loss import calculate_pairwise_affinity
from utils.dataset import cmap
from model.dino_txt_full_img import generate_pseudolabels_batch
from utils.sam import generate_contours_batch

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
                        text_emb_all, num_fg, temperature, min_max,
                        fastsam_model, sam_mask_generator, contour_method, dilation_size):
    """
    Visualize training sample following the same procedure as the training loop.

    Args:
        original_train_dataset: Original dataset (for getting original image and GT)
        train_dataset: Transformed dataset (for getting transformed image and target)
        model: Segmentation model
        index: Index of sample to visualize
        output_dir: Directory to save visualization
        text_emb_all: Precomputed text embeddings for all classes [num_all_classes, D]
        num_fg: Number of foreground classes
        temperature: Pseudolabel softmax temperature
        min_max: Whether pseudolabels are min-max scaled per channel
        fastsam_model: FastSAM model instance
        sam_mask_generator: SAM automatic mask generator instance (can be None)
        contour_method: Contour source, as in the training loop
        dilation_size: Contour max-filter size, as in the training loop
    """
    device = next(model.parameters()).device

    # Get original image and ground truth
    img, gt_mask = original_train_dataset[index]
    transformed_img, target, present = train_dataset[index]
    gt_mask = original_train_dataset.decode_target(gt_mask)

    model.eval()
    transformed_img_batch = transformed_img.unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through model to get segmentation and dino.txt patch tokens
        model_outputs = model(transformed_img_batch)
        segmentations = model_outputs['seg']  # [1, C, H, W]
        size = segmentations.shape[-2:]

        pseudolabel_probs_vis = generate_pseudolabels_batch(
            model_outputs['dinotxt'], present.unsqueeze(0).to(device), text_emb_all, num_fg,
            size, temperature, min_max
        )[0]  # [num_classes, H_seg, W_seg]

        sam_contours_x_batch, sam_contours_y_batch = generate_contours_batch(
            contour_method, transformed_img_batch, target.unsqueeze(0),
            train_dataset.denormalize, size, device,
            sam_mask_generator=sam_mask_generator, fastsam_model=fastsam_model
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
    fig, axes = plt.subplots(7, 2, figsize=(8, 28))
    
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
    w_x_vis = sam_contours_x_tensor.squeeze(0).cpu().numpy() if contour_method == 'color_diff' else calculate_pairwise_affinity(sam_contours_x_tensor, dilation_size).squeeze(0).cpu().numpy()
    w_y_vis = sam_contours_y_tensor.squeeze(0).cpu().numpy() if contour_method == 'color_diff' else calculate_pairwise_affinity(sam_contours_y_tensor, dilation_size).squeeze(0).cpu().numpy()
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
    output_resized = F.interpolate(
        output.unsqueeze(0), size=(img.size[1], img.size[0]), mode='bilinear', align_corners=False
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