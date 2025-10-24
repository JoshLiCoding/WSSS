import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.loss import calculate_pairwise_affinity
from utils.dataset import voc_cmap

def visualize_soft_probabilities(logits):
    probabilities = logits.softmax(dim=0).detach().numpy()
    num_classes, _, _ = probabilities.shape

    if not num_classes == 21:
        print('vis only supports 21 colours')
        return
    
    # Use VOC color palette instead of tab20
    voc_colors = voc_cmap(normalized=True)  # Get normalized RGB values [0,1]
    # VOC palette has 256 colors, we need the first 21 (classes 0-20)
    colors_array = voc_colors[:21]  # Shape: [21, 3]

    # Perform element-wise multiplication: (C, H, W, 1) * (C, 1, 1, 3) -> (C, H, W, 3)
    probabilities_expanded = np.expand_dims(probabilities, axis=-1)
    colors_reshaped = colors_array[:, np.newaxis, np.newaxis, :]
    weighted_colors = probabilities_expanded * colors_reshaped
    
    soft_vis = np.sum(weighted_colors, axis=0)
    soft_vis = np.clip(soft_vis*255, 0, 255).astype(np.uint8)
    soft_vis_image = Image.fromarray(soft_vis)
    return soft_vis_image

def vis_train_sample_img(voc_train_dataset, train_dataset, model, index, distance_transform, output_dir='.'):
    device = next(model.parameters()).device
    
    img, gt_mask = voc_train_dataset[index]
    transformed_img, pseudolabel_logits, sam_contours_x, sam_contours_y = train_dataset[index]

    gt_mask = voc_train_dataset.decode_target(gt_mask)

    model.eval()
    transformed_img = transformed_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(transformed_img)[0].cpu()
    transformed_img = train_dataset.denormalize(transformed_img[0].cpu()).permute(1, 2, 0)

    soft_pseudolabels = np.array(visualize_soft_probabilities(pseudolabel_logits / 0.05)) # temperature = 0.05
    pseudolabels_vis = pseudolabel_logits.argmax(0).numpy().astype(np.uint8)
    pseudolabels_vis = Image.fromarray(pseudolabels_vis)
    pseudolabels_vis = voc_train_dataset.decode_target(pseudolabels_vis)

    soft_output = np.array(visualize_soft_probabilities(output))
    output_vis = output.argmax(0).numpy().astype(np.uint8)
    output_vis = Image.fromarray(output_vis)
    output_vis = voc_train_dataset.decode_target(output_vis)
    
    fig, axes = plt.subplots(8, 2, figsize=(8, 24))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    
    axes[0, 1].imshow(gt_mask)
    axes[0, 1].set_title('GT mask')

    axes[1, 0].imshow(transformed_img)
    axes[1, 0].set_title('Transformed Image')

    axes[1, 1].axis('off')

    axes[2, 0].imshow(soft_pseudolabels)
    axes[2, 0].set_title('Soft Pseudolabels')

    axes[2, 1].imshow(pseudolabels_vis)
    axes[2, 1].set_title('Pseudolabels')

    axes[3, 0].imshow(sam_contours_x, cmap='grey')
    axes[3, 0].set_title('SAM Contours (horizontal)')

    axes[3, 1].imshow(sam_contours_y, cmap='grey')
    axes[3, 1].set_title('SAM Contours (vertical)')

    axes[4, 0].imshow(calculate_pairwise_affinity(sam_contours_x.unsqueeze(0), distance_transform).squeeze(0), cmap='grey')
    axes[4, 0].set_title('SAM Distance Field (horizontal)')

    axes[4, 1].imshow(calculate_pairwise_affinity(sam_contours_y.unsqueeze(0), distance_transform).squeeze(0), cmap='grey')
    axes[4, 1].set_title('SAM Distance Field (vertical)')

    axes[5, 0].imshow(soft_output)
    axes[5, 0].set_title('Soft Model Output')

    axes[5, 1].imshow(output_vis)
    axes[5, 1].set_title('Hard Model Output')
    
    H, W = output_vis.shape[:2]
    expanded_sam_contours_x = np.zeros((H, W), dtype=np.uint8)
    expanded_sam_contours_x[:, :W-1] = sam_contours_x
    axes[6, 0].imshow(expanded_sam_contours_x, alpha=0.5)
    axes[6, 0].imshow(soft_output, alpha=0.5)
    axes[6, 0].set_title('Soft Model Output & SAM Contours (horizontal)')

    expanded_sam_contours_y = np.zeros((H, W), dtype=np.uint8)
    expanded_sam_contours_y[:H-1, :] = sam_contours_y
    axes[6, 1].imshow(expanded_sam_contours_y, alpha=0.5)
    axes[6, 1].imshow(soft_output, alpha=0.5)
    axes[6, 1].set_title('Soft Model Output & SAM Contours (vertical)')

    axes[7, 0].imshow(expanded_sam_contours_x, alpha=0.5)
    axes[7, 0].imshow(output_vis, alpha=0.5)
    axes[7, 0].set_title('Hard Model Output & SAM Contours (horizontal)')

    axes[7, 1].imshow(expanded_sam_contours_y, alpha=0.5)
    axes[7, 1].imshow(output_vis, alpha=0.5)
    axes[7, 1].set_title('Hard Model Output & SAM Contours (vertical)')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'visualization_sample_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def vis_val_sample_img(voc_val_dataset, val_dataset, model, index, output_dir='.'):
    device = next(model.parameters()).device
    
    img, gt_mask = voc_val_dataset[index]
    transformed_img, _ = val_dataset[index]

    gt_mask = voc_val_dataset.decode_target(gt_mask)
    
    model.eval()
    transformed_img = transformed_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(transformed_img)[0].cpu()
    
    soft_output = np.array(visualize_soft_probabilities(output))

    output_vis = output.argmax(0).numpy().astype(np.uint8)
    output_vis = Image.fromarray(output_vis)
    output_vis = voc_val_dataset.decode_target(output_vis)
    
    # Resize model output to original image size
    output_resized = output.unsqueeze(0)
    output_resized = torch.nn.functional.interpolate(
        output_resized, size=(img.size[1], img.size[0]), mode='bilinear', align_corners=False
    )[0]
    output_resized_vis = output_resized.argmax(0).numpy().astype(np.uint8)
    output_resized_vis = Image.fromarray(output_resized_vis)
    output_resized_vis = voc_val_dataset.decode_target(output_resized_vis)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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

    axes[0, 1].axis('off')

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