"""Stage 1: cache the DINOv2 pseudo-labels and the mask boundaries for the train split.

Both caches are keyed by image name, so this is resumable and skips whatever already exists.
main.py calls the same two functions, so running this first is optional -- it just keeps the
heavy one-off generation in its own job.
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from model.dinov2_cam import ensure_pseudolabels
from utils.boundaries import ensure_boundaries
from utils.dataset import build_dataset


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = build_dataset(cfg, 'train')
    ensure_pseudolabels(cfg, train_dataset, cfg.pseudolabel.dir, device)
    ensure_boundaries(cfg, train_dataset, cfg.boundary.dir, device)


if __name__ == "__main__":
    main()
