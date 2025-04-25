#!/usr/bin/env python3
# coding: utf-8

# Force non-interactive backend for Matplotlib and disable DISPLAY for headless
import matplotlib
matplotlib.use('Agg')
import os
os.environ['DISPLAY'] = ''

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

# Quantitative evaluation
input_dir = os.path.join('data', 'PU1K', 'test', 'input_256', 'input_256')
gt_dir    = os.path.join('data', 'PU1K', 'test', 'input_256', 'gt_1024')

chamfers, dcds, hausdorffs, times = [], [], [], []
for fname in tqdm(os.listdir(input_dir), desc='Evaluating'):
    inp = torch.tensor(load_xyz_file(os.path.join(input_dir, fname)), dtype=torch.float32).to(device)
    gt_  = torch.tensor(load_xyz_file(os.path.join(gt_dir, fname)), dtype=torch.float32).to(device)

    start = time.time()
    with torch.no_grad():
        out = model(inp)
    elapsed = time.time() - start

    dcd, _, cd = density_chamfer_dist(out.unsqueeze(0), gt_.unsqueeze(0), alpha=1000, n_lambda=0.5)
    hd = hausdorff_loss(out, gt_)

    chamfers.append(cd.item())
    dcds.append(dcd.item())
    hausdorffs.append(hd.item())
    times.append(elapsed)

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
