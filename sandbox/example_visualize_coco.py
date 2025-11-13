"""
Example script showing how to use the COCO pseudolabel visualization.

Usage examples:
"""

from visualize_coco_pseudolabels import visualize_sample_simple, visualize_coco_sample
from utils.dataset import COCOSegmentation

# Example 1: Simple visualization with default paths
if __name__ == '__main__':
    # Method 1: Use the simple function
    visualize_sample_simple(
        coco_root='coco',  # Path to COCO dataset root
        pseudolabels_dir='pseudolabels_coco',  # Directory with pseudolabel .npy files
        index=4,  # Sample index to visualize
        image_set='train',  # 'train' or 'val'
        year='2014'  # '2014' or '2017'
    )
    
    # Method 2: Use the full function with more control
    # coco_dataset = COCOSegmentation(
    #     root='coco',
    #     year='2014',
    #     image_set='train',
    #     n_images=-1
    # )
    # 
    # visualize_coco_sample(
    #     coco_dataset,
    #     index=0,
    #     pseudolabels_dir='pseudolabels_coco',
    #     output_path='my_custom_output.png'
    # )
    
    # Method 3: Use command line
    # python visualize_coco_pseudolabels.py --coco_root coco --pseudolabels_dir pseudolabels_coco --index 0
    # python visualize_coco_pseudolabels.py --coco_root coco --pseudolabels_dir pseudolabels_coco --index 0 --n_samples 5

