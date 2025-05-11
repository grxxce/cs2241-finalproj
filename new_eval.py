#!/usr/bin/env python3
# coding: utf-8

# Force non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')
import os
os.environ['DISPLAY'] = ''
import matplotlib.pyplot as plt
from utils.viz import viz_many_mpl, point_cloud_three_views
import sys
import time
import logging
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb

from utils.data import load_xyz_file
from utils.chamfer_distance import density_chamfer_dist
from utils.losses import hausdorff_loss

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

# Load experiment config
conf = OmegaConf.load(os.path.join('conf', 'config.yaml'))
log.info(f"Experiment: {conf.name}")
wandb.init(
    project=conf.name,
    config=OmegaConf.to_container(conf, resolve=True)
)

# Set paths
run_id = '2025-05-09T02:04:06.684641'
# run_id = '2025-05-09T01:59:20.532181' done
# run_id = '2025-05-09T01:52:51.999337' done
# run_id = '2025-05-09T01:50:28.252384' done

# run_id = '2025-05-09T02:29:43.845555' done
# run_id = '2025-05-09T02:33:14.232064' done
# run_id = '2025-05-09T02:36:44.373272' done
# run_id = '2025-05-09T02:44:27.350831' done

# run_id = '2025-05-09T10:38:13.120407' done
# run_id = '2025-05-09T13:07:49.844928'

pred_dir = f'../Grad-PU/output/{run_id}/test/4X-256'  # Your predictions directory
gt_dir = '../Grad-PU/data/PU1K/test/input_256/gt_1024'        # Ground truth directory

# Prepare output folders for visualizations
qual_dir = os.path.join('images', 'qualitative', 'small', run_id)
views_dir = os.path.join('images', 'views', 'small', run_id)
os.makedirs(qual_dir, exist_ok=True)
os.makedirs(views_dir, exist_ok=True)

# Initialize metrics lists
chamfers, dcds, hausdorffs, times = [], [], [], []

# Get the list of files to evaluate
# Assuming the filenames in pred_dir match those in gt_dir
pred_files = os.listdir(pred_dir)
log.info(f"Found {len(pred_files)} prediction files to evaluate")

# Check if files exist in ground truth directory
gt_files = os.listdir(gt_dir)
common_files = [f for f in pred_files if f in gt_files]
if len(common_files) < len(pred_files):
    log.warning(f"Only {len(common_files)} files have matching ground truth")
    pred_files = common_files

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log.info(f"Using device: {device}")

for fname in tqdm(pred_files, desc='Evaluating'):
    # Load predicted and ground truth point clouds
    pred_path = os.path.join(pred_dir, fname)
    gt_path = os.path.join(gt_dir, fname)
    
    # Skip if either file doesn't exist
    if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
        log.warning(f"Skipping {fname} - file not found in both directories")
        continue
    
    # Load and move to device
    pred = torch.tensor(load_xyz_file(pred_path), dtype=torch.float32).to(device)
    gt = torch.tensor(load_xyz_file(gt_path), dtype=torch.float32).to(device)
    
    # Timing is now just for visualization
    t0 = time.time()
    
    # Compute metrics
    dcd, _, cd = density_chamfer_dist(pred.unsqueeze(0), gt.unsqueeze(0), alpha=1000, n_lambda=0.5)
    hd = hausdorff_loss(pred, gt)
    
    dt = time.time() - t0
    times.append(dt)

    print(f"cd: {cd.item()}, dcd: {dcd.item()}, hd: {hd.item()}")
    
    chamfers.append(cd.item())
    dcds.append(dcd.item())
    hausdorffs.append(hd.item())
    
    # 1) Save the 3D scatter comparison
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Load input point cloud if available
    input_dir = os.path.join('data', 'PU1K', 'test', 'input_256', 'input_256')
    input_path = os.path.join(input_dir, fname)
    if os.path.exists(input_path):
        inp = torch.tensor(load_xyz_file(input_path), dtype=torch.float32).cpu().numpy()
        viz_many_mpl(
            [inp, gt.cpu().numpy(), pred.cpu().numpy()],
            ax=ax
        )
    else:
        # Just show GT and prediction if input not available
        viz_many_mpl(
            [gt.cpu().numpy(), pred.cpu().numpy()],
            ax=ax
        )
    
    qual_path = os.path.join(qual_dir, fname.replace('.xyz', '.png'))
    fig.savefig(qual_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved qualitative viz → {qual_path}")
    
    # 2) Save the multi-view "screenshots" as a single image
    views_img = point_cloud_three_views(pred.cpu().numpy())
    mv_path = os.path.join(views_dir, fname.replace('.xyz', '_views.png'))
    plt.imsave(mv_path, views_img, cmap='gray', format='png')
    log.info(f"Saved multi‐view viz → {mv_path}")

# Aggregate metrics
metrics = {
    'chamfer_mse': float(np.mean(chamfers) * 1e3),
    'density_chamfer': float(np.mean(dcds)),
    'hausdorff_loss': float(np.mean(hausdorffs) * 1e3),
    'computation_time_ms': float(np.mean(times) * 1e3)
}
log.info(f"Metrics: {metrics}")
wandb.log(metrics)

# Save results
results = OmegaConf.create({**metrics, 'files_evaluated': len(chamfers)})
out_dir = 'results'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_out = os.path.join(out_dir, f"results-{run_id}-{timestamp}.yaml")
OmegaConf.save(results, file_out)
log.info(f"Saved results: {file_out}")

print("\n" + "="*50)
print(f"EVALUATION RESULTS ({len(chamfers)} files):")
print(f"Chamfer MSE:      {metrics['chamfer_mse']:.4f}")
print(f"Density Chamfer:  {metrics['density_chamfer']:.4f}")
print(f"Hausdorff Loss:   {metrics['hausdorff_loss']:.4f}")
print("="*50 + "\n")

wandb.finish()