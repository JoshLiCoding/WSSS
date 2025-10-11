import torch

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