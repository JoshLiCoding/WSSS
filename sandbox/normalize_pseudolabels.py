#!/usr/bin/env python3
"""
Normalize pseudolabels using min-max normalization + simplex projection.
Processes all pseudolabel files in pseudolabels_2 and saves to pseudolabels_2_normalized.
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse

def project_rows_to_simplex(X):
    """
    Projects each row of X to the probability simplex (L2 projection).
    Implements the algorithm from Wang & Carreira-Perpinan.
    
    Args:
        X: numpy array of shape (N, C) where each row is a point
    Returns:
        Xp: array of shape (N, C) with each row projected to simplex
    """
    Xp = np.copy(X)
    N, C = Xp.shape
    for i in range(N):
        v = Xp[i]
        if v.sum() == 1 and np.all(v >= 0):
            continue
        # sort descending
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, C+1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        w = np.maximum(v - theta, 0.0)
        Xp[i] = w
    return Xp

def min_max_normalize(X, eps=1e-12):
    """
    Min-max normalize each column (class) to [0, 1].
    
    Args:
        X: numpy array of shape (N, C) where columns are normalized
    Returns:
        Xc: array of shape (N, C) with each column in [0, 1]
    """
    Xc = X.copy()
    low = np.min(Xc, axis=0)
    high = np.max(Xc, axis=0)
    span = np.maximum(high - low, eps)
    Xc = (Xc - low[np.newaxis, :]) / span[np.newaxis, :]
    return Xc

def normalize_pseudolabels(pseudolabels, num_iterations=5, temperature=0.05):
    """
    Apply softmax with temperature, then iterative normalization: min-max + simplex projection.
    
    Args:
        pseudolabels: numpy array of shape (H, W, C)
        num_iterations: number of iterations
        temperature: temperature for softmax (default: 0.05)
    Returns:
        normalized pseudolabels of shape (H, W, C)
    """
    H, W, C = pseudolabels.shape
    
    # Apply softmax with temperature along channel dimension
    pseudolabels_scaled = pseudolabels / temperature
    exp_vals = np.exp(pseudolabels_scaled)
    pseudolabels_np = exp_vals / np.sum(exp_vals, axis=2, keepdims=True)
    
    for i in range(num_iterations):
        # Reshape to (H*W, C) - each row is a pixel with C class probabilities
        pseudolabels_flat = pseudolabels_np.reshape(-1, C)
        # Min-max normalize each class (column)
        pseudolabels_flat = min_max_normalize(pseudolabels_flat)
        # Project each pixel (row) to probability simplex
        pseudolabels_flat = project_rows_to_simplex(pseudolabels_flat)
        # Reshape back to (H, W, C)
        pseudolabels_np = pseudolabels_flat.reshape(H, W, C)
    
    return pseudolabels_np

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Normalize pseudolabels with optional index range')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting index (inclusive, default: 0)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='Ending index (exclusive, default: all files)')
    parser.add_argument('--input_dir', type=str, default='pseudolabels_2',
                        help='Input directory (default: pseudolabels_2)')
    parser.add_argument('--output_dir', type=str, default='pseudolabels_2_normalized',
                        help='Output directory (default: pseudolabels_2_normalized)')
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='Number of normalization iterations (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Temperature for softmax (default: 0.05)')
    args = parser.parse_args()
    
    # Directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Find all pseudolabel files
    all_pseudolabel_files = sorted(input_dir.glob('pseudolabels_*.npy'))
    
    # Extract indices from filenames and filter by range
    file_index_pairs = []
    for f in all_pseudolabel_files:
        idx_str = f.name.replace('pseudolabels_', '').replace('.npy', '')
        try:
            idx = int(idx_str)
            file_index_pairs.append((idx, f))
        except ValueError:
            print(f"Warning: Could not parse index from {f.name}, skipping...")
            continue
    
    # Sort by index
    file_index_pairs.sort(key=lambda x: x[0])
    
    # Filter by start and end index
    if args.end_index is not None:
        file_index_pairs = [(idx, f) for idx, f in file_index_pairs 
                           if args.start_index <= idx < args.end_index]
    else:
        file_index_pairs = [(idx, f) for idx, f in file_index_pairs 
                           if idx >= args.start_index]
    
    pseudolabel_files = [f for idx, f in file_index_pairs]
    indices = [idx for idx, f in file_index_pairs]
    
    print(f"Total pseudolabel files in {input_dir}: {len(all_pseudolabel_files)}")
    print(f"Processing range: [{args.start_index}, {args.end_index if args.end_index else 'end'})")
    print(f"Files to process: {len(pseudolabel_files)}")
    if len(pseudolabel_files) > 0:
        print(f"Index range: {indices[0]} to {indices[-1]}")
    print(f"Output directory: {output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"Normalization: {args.num_iterations} iterations of (min-max + simplex projection)")
    print()
    
    # Process each file
    for pseudolabel_file in tqdm(pseudolabel_files, desc="Normalizing pseudolabels"):
        # Extract index from filename
        filename = pseudolabel_file.name
        idx = filename.replace('pseudolabels_', '').replace('.npy', '')
        
        # Load pseudolabels and class_indices
        pseudolabels = np.load(pseudolabel_file)
        class_indices_file = input_dir / f'class_indices_{idx}.npy'
        
        if not class_indices_file.exists():
            print(f"Warning: Missing class_indices file for {filename}, skipping...")
            continue
        
        class_indices = np.load(class_indices_file)
        
        # Normalize pseudolabels
        pseudolabels_normalized = normalize_pseudolabels(
            pseudolabels, 
            num_iterations=args.num_iterations,
            temperature=args.temperature
        )
        
        # Save normalized pseudolabels
        np.save(output_dir / f'pseudolabels_{idx}.npy', pseudolabels_normalized)
        
        # Copy class_indices (unchanged)
        shutil.copy(class_indices_file, output_dir / f'class_indices_{idx}.npy')
    
    print()
    print(f"✓ Normalization complete!")
    print(f"  Processed {len(pseudolabel_files)} files")
    print(f"  Saved to: {output_dir}")

if __name__ == '__main__':
    main()

