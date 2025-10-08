import os
import torch
from torch import nn
import numpy as np
from dinov3.data.transforms import make_classification_eval_transform

class DinoModel(nn.Module):
    def __init__(self):
        super(DinoModel, self).__init__()
        DINOV3_LOCATION = 'model/dinov3'
        self.model, self.tokenizer = torch.hub.load(DINOV3_LOCATION,
            'dinov3_vitl16_dinotxt_tet1280d20h24l', 
            source='local',
            weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'), 
            backbone_weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'))
        
    def forward(self, x):
        image_preprocess = make_classification_eval_transform()
        x = image_preprocess(x)
        image_class_tokens, image_patch_tokens, backbone_patch_tokens = self.model.encode_image_with_patch_tokens(x)
        return backbone_patch_tokens


image_preprocess = make_classification_eval_transform()
image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).cuda()
texts = ["a photo of a dog", "chair", "floor", "wall"]
class_names = ["dog", "chair", "floor", "wall"]
tokenized_texts_tensor = tokenizer.tokenize(texts).cuda()

with torch.autocast('cuda', dtype=torch.float):
    with torch.no_grad():
        image_class_tokens, image_patch_tokens, backbone_patch_tokens = model.encode_image_with_patch_tokens(image_tensor)
        text_features_aligned_to_patch = model.encode_text(tokenized_texts_tensor)[:, 1024:] # Part of text features that is aligned to patch features

B, P, D = image_patch_tokens.shape
H = W = int(P**0.5) 
x = image_patch_tokens.movedim(2, 1).unflatten(2, (H, W)).float()  # [B, D, H, W]
x = F.interpolate(x, size=(480, 640), mode="bicubic", align_corners=False)
x = F.normalize(x, p=2, dim=1)
y = F.normalize(text_features_aligned_to_patch.float(), p=2, dim=1)
per_patch_similarity_to_text = torch.einsum("bdhw,cd->bchw", x, y)
pred_idx = per_patch_similarity_to_text.argmax(1).squeeze(0)

def visualize_text_segmentation(pred_idx, class_names, img_pil, alpha=0.5):
    """
    Visualize per-pixel text similarity result as a color overlay.
    """
    # Convert to numpy
    pred_np = pred_idx.cpu().numpy() if torch.is_tensor(pred_idx) else np.array(pred_idx)
    H, W = pred_np.shape

    # Assign colors (tab10 colormap has 10 distinct colors)
    cmap = plt.cm.get_cmap("tab10", len(class_names))
    color_map = cmap(np.arange(len(class_names)))[:, :3]  # RGB only

    # Create RGB segmentation map
    seg_rgb = color_map[pred_np]  # [H, W, 3]

    # Convert original image to float [0,1]
    img_rgb = np.array(img_pil.convert("RGB")) / 255.0
    img_rgb = np.array(Image.fromarray((img_rgb * 255).astype(np.uint8)).resize((W, H))) / 255.0

    # Blend overlay
    overlay = (1 - alpha) * img_rgb + alpha * seg_rgb

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map[i], label=class_names[i])
        for i in range(len(class_names))
    ]
    plt.legend(handles=legend_patches, loc="upper right", fontsize=10)
    plt.title("Per-patch text similarity segmentation")
    plt.tight_layout()
    plt.show()

# Example usage
visualize_text_segmentation(pred_idx, class_names, img_pil)