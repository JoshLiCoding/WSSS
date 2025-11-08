# Foundation Models Define CRF Potentials in Weakly Supervised Semantic Segmentation

This repository implements losses using the CRF/Potts formulation, with dino.txt pseudolabels as unary potentials and SAM edges as pairwise potentials. The segmentation models are a custom decoder (DINO backbone -> self-attention blocks -> upsample 4x -> CNN blocks -> upsample 4x) or standard Deeplabv3(+).
