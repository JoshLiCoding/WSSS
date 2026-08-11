import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.loss import calculate_pairwise_affinity
from utils.pseudolabels import soft_probability_vis, soft_pseudolabels


def vis_train_sample_img(original_train_dataset, train_dataset, model, index, output_dir,
                        num_classes, temperature, contour_method):
    """
    Visualize training sample following the same procedure as the training loop.

    Args:
        original_train_dataset: Original dataset (for getting original image and GT)
        train_dataset: Augmented dataset (image, target, cached pseudolabels and boundaries)
        model: Segmentation model
        index: Index of sample to visualize
        output_dir: Directory to save visualization
        num_classes: Number of classes
        temperature: Pseudo-label softmax temperature
        contour_method: Name of the method the cached boundaries came from (for titles)
    """
    device = next(model.parameters()).device

    # Get original image and ground truth
    img, gt_mask = original_train_dataset[index]
    transformed_img, target, pseudolabel_sim, present, edges = train_dataset[index]
    gt_mask = original_train_dataset.decode_target(gt_mask)
    transformed_gt_mask = original_train_dataset.decode_target(target.numpy().astype(np.uint8))

    model.eval()
    transformed_img_batch = transformed_img.unsqueeze(0).to(device)

    with torch.no_grad():
        segmentations = model(transformed_img_batch)  # [1, C, H, W]
        _, _, H_seg, W_seg = segmentations.shape

        pseudolabel_probs_vis = soft_pseudolabels(
            pseudolabel_sim.unsqueeze(0).to(device), present.unsqueeze(0).to(device), temperature
        )[0]

        contours_x = edges[0, :, :-1].float().numpy()  # [H, W-1]
        contours_y = edges[1, :-1, :].float().numpy()  # [H-1, W]

    # Prepare visualization data
    transformed_img_vis = train_dataset.denormalize(transformed_img_batch[0].cpu()).permute(1, 2, 0)
    segmentation_vis = segmentations[0]  # [C, H, W]

    # Visualize pseudolabel probabilities
    soft_pseudolabels_vis = soft_probability_vis(pseudolabel_probs_vis)
    pseudolabels_vis = pseudolabel_probs_vis.argmax(0).cpu().numpy().astype(np.uint8)
    pseudolabels_vis = Image.fromarray(pseudolabels_vis)
    pseudolabels_vis = original_train_dataset.decode_target(pseudolabels_vis)

    # Visualize segmentation output
    soft_output = soft_probability_vis(segmentation_vis.softmax(dim=0))
    output_vis = segmentation_vis.argmax(0).cpu().numpy().astype(np.uint8)
    output_vis = Image.fromarray(output_vis)
    output_vis = original_train_dataset.decode_target(output_vis)

    # Create visualization
    fig, axes = plt.subplots(6, 2, figsize=(8, 24))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')

    axes[0, 1].imshow(gt_mask)
    axes[0, 1].set_title('GT mask')

    axes[1, 0].imshow(transformed_img_vis)
    axes[1, 0].set_title('Transformed Image')

    axes[1, 1].imshow(transformed_gt_mask)
    axes[1, 1].set_title('Transformed GT mask')

    axes[2, 0].imshow(soft_pseudolabels_vis)
    axes[2, 0].set_title('Soft Pseudolabels')

    axes[2, 1].imshow(pseudolabels_vis)
    axes[2, 1].set_title('Pseudolabels')

    contour_title = f'{contour_method.upper()} Contours'
    axes[3, 0].imshow(contours_x, cmap='gray', vmin=0, vmax=1)
    axes[3, 0].set_title(f'{contour_title} (horizontal)')

    axes[3, 1].imshow(contours_y, cmap='gray', vmin=0, vmax=1)
    axes[3, 1].set_title(f'{contour_title} (vertical)')

    contours_x_tensor = edges[0:1, :, :-1].float().to(device)  # [1, H, W-1]
    contours_y_tensor = edges[1:2, :-1, :].float().to(device)  # [1, H-1, W]
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
        output = model(transformed_img)[0].cpu()

    soft_output = soft_probability_vis(output.softmax(dim=0))

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
    soft_output_resized = soft_probability_vis(output_resized.softmax(dim=0))

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