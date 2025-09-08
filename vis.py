import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_soft_probabilities(logits):
    probabilities = logits.softmax(dim=0).detach().numpy()
    num_classes, height, width = probabilities.shape

    if not num_classes == 21:
        print('vis only supports 21 colours')
        return
    
    cmap = plt.get_cmap('tab20')
    colors_tab20 = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    additional_color_rgb = (0.3, 0.3, 0.3)
    colors_array = np.array(colors_tab20 + [additional_color_rgb])

    # Perform element-wise multiplication: (C, H, W, 1) * (C, 1, 1, 3) -> (C, H, W, 3)
    probabilities_expanded = np.expand_dims(probabilities, axis=-1)
    colors_reshaped = colors_array[:, np.newaxis, np.newaxis, :]
    weighted_colors = probabilities_expanded * colors_reshaped
    
    soft_vis = np.sum(weighted_colors, axis=0)
    soft_vis = np.clip(soft_vis*255, 0, 255).astype(np.uint8)
    soft_vis_image = Image.fromarray(soft_vis)
    return soft_vis_image

def vis_sample_img(voc_train_dataset, train_dataset, deeplabv3, index=0, output_dir='.'):
    device = next(deeplabv3.parameters()).device
    
    img, gt_mask = voc_train_dataset[index]
    transformed_img, pseudolabel_logits, sam_contour = train_dataset[index]
    voc_palette = gt_mask.getpalette()
    
    psuedolabel_vis = pseudolabel_logits.softmax(0).argmax(0).numpy().astype(np.uint8)
    psuedolabel_vis = Image.fromarray(psuedolabel_vis)
    psuedolabel_vis.putpalette(voc_palette)
    
    deeplabv3.eval()
    transformed_img = transformed_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplabv3(transformed_img)['out'][0]
        
    output_vis = output.cpu().softmax(0).argmax(0).numpy().astype(np.uint8)
    output_vis = Image.fromarray(output_vis)
    output_vis.putpalette(voc_palette)
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 18))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    
    axes[0, 1].imshow(gt_mask)
    axes[0, 1].set_title('GT mask')

    axes[1, 0].imshow(visualize_soft_probabilities(pseudolabel_logits))
    axes[1, 0].set_title('Soft Pseudolabel')

    axes[1, 1].imshow(psuedolabel_vis)
    axes[1, 1].set_title('Hard Pseudolabel')

    axes[1, 2].imshow(sam_contour, cmap='grey')
    axes[1, 2].set_title('SAM contour')
    
    axes[2, 0].imshow(visualize_soft_probabilities(output.cpu()))
    axes[2, 0].set_title('Soft Output')
    
    axes[2, 1].imshow(output_vis)
    axes[2, 1].set_title('Hard Output')
    
    axes[3, 0].imshow(sam_contour, cmap='grey')
    axes[3, 0].imshow(visualize_soft_probabilities(pseudolabel_logits), alpha=0.8)
    axes[3, 0].set_title('Overlayed Soft Pseduolabel')

    axes[3, 1].imshow(sam_contour, cmap='grey')
    axes[3, 1].imshow(visualize_soft_probabilities(output.cpu()), alpha=0.8)
    axes[3, 1].set_title('Overlayed Soft Output')

    axes[0, 2].remove()
    axes[2, 2].remove()
    axes[3, 2].remove()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'visualization_sample_{index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as '{save_path}'")

def vis_training_loss(NUM_EPOCHS, epoch_total_losses, epoch_cce_main_losses, epoch_potts_main_losses, output_dir='.'):
    # Graph 1: Total Loss
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, epoch_total_losses, label='Total Loss', color='blue', linewidth=2)
    plt.title('Total Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'total_loss.png'), dpi=300, bbox_inches='tight')

    # Graph 2: Individual Loss Components
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, epoch_cce_main_losses, label='CCE Main Loss', color='red', linestyle='--')
    plt.plot(epochs, epoch_potts_main_losses, label='Potts Main Loss', color='green', linestyle='-')

    plt.title('Individual Loss Components Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'individual_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training loss visualizations saved")