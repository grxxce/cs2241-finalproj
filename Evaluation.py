#!/usr/bin/env python3
# coding: utf-8

# Force non-interactive backend for Matplotlib and disable DISPLAY for headless
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
from pugcn_lib import PUGCN

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

# Load latest checkpoint
models_dir = 'trained-models'
latest = sorted(os.listdir(models_dir))[-1]
ckpt_path = os.path.join(models_dir, latest, f"ckpt_epoch_{conf.train_config.epochs}.pt")
log.info(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path)

# Build model
model = PUGCN(**conf.model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

input_dir = os.path.join('data','PU1K','test','input_256','input_256')
gt_dir    = os.path.join('data','PU1K','test','input_256','gt_1024')

# prepare output folders
qual_dir  = os.path.join('images','qualitative', latest)
views_dir = os.path.join('images','views',       latest)
os.makedirs(qual_dir,  exist_ok=True)
os.makedirs(views_dir, exist_ok=True)

chamfers, dcds, hausdorffs, times = [], [], [], []

for fname in tqdm(os.listdir(input_dir), desc='Evaluating'):
    # load and move to device
    inp = torch.tensor(load_xyz_file(os.path.join(input_dir, fname)), dtype=torch.float32).to(device)
    gt  = torch.tensor(load_xyz_file(os.path.join(gt_dir,    fname)), dtype=torch.float32).to(device)

    # inference + timing
    t0 = time.time()
    with torch.no_grad():
        out = model(inp)
    dt = time.time() - t0
    times.append(dt)

    # compute metrics
    dcd, _, cd = density_chamfer_dist(out.unsqueeze(0), gt.unsqueeze(0), alpha=1000, n_lambda=0.5)
    hd         = hausdorff_loss(out, gt)
    chamfers.append(cd.item())
    dcds.append(dcd.item())
    hausdorffs.append(hd.item())

    # 1) Save the 3D scatter comparison
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, projection='3d')
    viz_many_mpl(
        [inp.cpu().numpy(), gt.cpu().numpy(), out.cpu().numpy()],
        ax=ax
    )
    qual_path = os.path.join(qual_dir, fname.replace('.xyz', '.png'))
    fig.savefig(qual_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved qualitative viz → {qual_path}")

    # 2) Save the multi-view “screenshots” as a single image
    views_img = point_cloud_three_views(out.cpu().numpy())
    mv_path   = os.path.join(views_dir, fname.replace('.xyz', '_views.png'))
    plt.imsave(mv_path, views_img, cmap='gray', format='png')
    log.info(f"Saved multi‐view viz → {mv_path}")

# Aggregate metrics
metrics = {
    'chamfer_mse': float(np.mean(chamfers) * 1e3),
    'density_chamfer': float(np.mean(dcds)),
    'hausdorff_loss': float(np.mean(hausdorffs) * 1e3),
    'avg_inference_time_ms': float(np.mean(times) * 1e3)
}
log.info(f"Metrics: {metrics}")
wandb.log(metrics)

# Save results
results = OmegaConf.create({**metrics, 'epochs_trained': checkpoint['epoch'], 'model_size': checkpoint.get('model_size')})
out_dir = 'results'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
file_out = os.path.join(out_dir, f"results-{latest}-ep{checkpoint['epoch']}.yaml")
OmegaConf.save(results, file_out)
log.info(f"Saved results: {file_out}")

wandb.finish()
