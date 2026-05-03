import torch
import torch.nn.functional as F

def test_time_augmentation_inference(model, image, target_size, scales=[1.0], base_size=(224, 224), device='cuda'):
    """
    Perform test-time augmentation by running inference at multiple scales with horizontal flips and aggregating outputs.
    When base_size is set, input is interpolated once to (base_size * scale) per scale; otherwise uses image size * scale.

    Args:
        model: The segmentation model
        image: Input image tensor [C, H, W] (already normalized, original resolution)
        target_size: Target size (H, W) to resize all outputs to
        scales: List of scale factors to apply
        base_size: (H, W) base size; input is resized to (base_size[0]*s, base_size[1]*s) for each scale s
        device: Device to run inference on

    Returns:
        Aggregated segmentation logits [1, num_classes, H, W]
    """
    (bh, bw) = base_size
    all_outputs = []

    with torch.no_grad():
        for scale in scales:
            size = (int(bh * scale), int(bw * scale))
            scaled_image = F.interpolate(
                image.unsqueeze(0), size=size,
                mode='bilinear', align_corners=False
            )
            
            # Run inference on original orientation
            output = model(scaled_image)
            segmentation = output['seg']  # [1, num_classes, H_scaled, W_scaled]
            
            # Resize output back to target size
            segmentation_resized = F.interpolate(
                segmentation,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            all_outputs.append(segmentation_resized)
            
            # Run inference on horizontally flipped image
            scaled_image_flipped = torch.flip(scaled_image, dims=[-1])  # Flip along width dimension
            output_flipped = model(scaled_image_flipped)
            segmentation_flipped = output_flipped['seg']  # [1, num_classes, H_scaled, W_scaled]
            
            # Flip the output back to original orientation
            segmentation_flipped = torch.flip(segmentation_flipped, dims=[-1])
            
            # Resize flipped output back to target size
            segmentation_flipped_resized = F.interpolate(
                segmentation_flipped,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            all_outputs.append(segmentation_flipped_resized)
    
    # Average all outputs
    aggregated_output = torch.stack(all_outputs).mean(dim=0)
    
    return aggregated_output

def update_miou(predictions, targets, intersection_counts, union_counts, num_classes, ignore_index=255):
    predictions = torch.nn.functional.interpolate(
        predictions, size=targets.shape[-2:], mode='bilinear', align_corners=False
    )
    predictions = torch.argmax(predictions, dim=1) # (B, H, W)
    
    # create mask for valid pixels (not ignore_index)
    valid_mask = (targets != ignore_index)
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
             
        pred_mask = (predictions == cls) & valid_mask
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).float().sum().item()
        union = (pred_mask | target_mask).float().sum().item()
        
        intersection_counts[cls] += intersection
        union_counts[cls] += union