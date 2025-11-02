"""Test script for COCOSegmentation dataset"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import COCOSegmentation
from PIL import Image

def visualize_sample(img, mask, idx=0):
    """Visualize image and mask"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert mask to numpy array
    mask_array = np.array(mask)
    
    # Show original image
    axes[0].imshow(img)
    axes[0].set_title(f'Sample {idx} - Original Image')
    axes[0].axis('off')
    
    # Use dataset's colormap to decode mask
    colored_mask = COCOSegmentation.decode_target(mask_array)
    
    # Show colored mask
    axes[1].imshow(colored_mask)
    axes[1].set_title('Semantic Mask')
    axes[1].axis('off')
    
    # Show overlay
    img_array = np.array(img)
    overlay = img_array.copy()
    
    # Create a semi-transparent overlay for non-background pixels
    non_bg_mask = mask_array > 0
    overlay[non_bg_mask] = overlay[non_bg_mask] * 0.5 + colored_mask[non_bg_mask] * 0.5
    
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Image with Mask Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'coco_sample_{idx}_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to coco_sample_{idx}_visualization.png")
    plt.close()

def test_coco_dataset():
    # Example usage - update these paths to match your COCO dataset location
    root = 'coco'  # Update this to your COCO dataset root
    # Expected structure:
    # coco/
    #   ├── train2014/
    #   ├── val2014/
    #   └── annotations/
    #       ├── instances_train2014.json
    #       └── instances_val2014.json
    
    try:
        # Test train dataset
        print("Loading COCO train dataset...")
        coco_train = COCOSegmentation(
            root=root,
            year='2014',
            image_set='train',
            n_images=10  # Test with just 10 images
        )
        print(f"✓ Loaded {len(coco_train)} training images")
        print(f"✓ Number of classes: {coco_train.num_classes}")
        
        # Test getting an item
        print("\nTesting __getitem__...")
        img, mask = coco_train[0]
        print(f"✓ Image size: {img.size}, Mask size: {mask.size}")
        print(f"✓ Image mode: {img.mode}, Mask mode: {mask.mode}")
        
        # Check mask values
        mask_array = np.array(mask)
        unique_values = np.unique(mask_array)
        print(f"✓ Unique mask values: {unique_values}")
        print(f"✓ Number of unique classes in mask: {len(unique_values)}")
        
        # Visualize a few samples
        print("\nCreating visualizations...")
        n_vis = min(3, len(coco_train))
        for i in range(n_vis):
            img, mask = coco_train[i]
            visualize_sample(img, mask, idx=i)
            print(f"  ✓ Visualized sample {i}")
        
        # Test val dataset
        print("\nLoading COCO val dataset...")
        coco_val = COCOSegmentation(
            root=root,
            year='2014',
            image_set='val',
            n_images=5
        )
        print(f"✓ Loaded {len(coco_val)} validation images")
        
        # Visualize validation sample
        if len(coco_val) > 0:
            img_val, mask_val = coco_val[0]
            visualize_sample(img_val, mask_val, idx='val_0')
        
        print("\n✅ All tests passed!")
        
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        print("\nPlease update the 'root' path in this script to point to your COCO dataset")
        print("\nDownload COCO from: http://cocodataset.org/#download")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_coco_dataset()
