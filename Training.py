#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
import logging
from datetime import datetime

import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader as PyGLoader

import torch._dynamo
# Suppress Dynamo errors to fall back to eager ops
torch._dynamo.config.suppress_errors = True

import torchinfo
from omegaconf import OmegaConf

import wandb

from utils import model_size
from utils.losses import chamfer_dist_repulsion, density_chamfer_dist
from utils.data import PCDDataset
from pugcn_lib import PUGCN


def train_one_epoch(model, loader, loss_fn, optimizer, gamma, device, k_loss):
    model.train()
    total_loss = 0.0
    total_prop = 0.0
    for i, batch in enumerate(loader):
        p, q = batch.pos_s.to(device), batch.pos_t.to(device)
        p_batch = getattr(batch, 'pos_s_batch', None)
        q_batch = getattr(batch, 'pos_t_batch', None)
        if p_batch is not None: p_batch = p_batch.to(device)
        if q_batch is not None: q_batch = q_batch.to(device)

        optimizer.zero_grad()
        pred = model(p, batch=p_batch)
        pred, _ = to_dense_batch(pred, q_batch)
        gt, _   = to_dense_batch(q, q_batch)

        if loss_fn == "dcd":
            dcd, _, _ = density_chamfer_dist(pred, gt)
            loss = dcd.mean()
        elif loss_fn == "cd":
            _, _, cd_t = density_chamfer_dist(pred, gt)
            loss = cd_t.mean()
        elif loss_fn == "cd_rep":
            loss, prop = chamfer_dist_repulsion(pred, gt, k=k_loss, return_proportion=True)
            total_prop += prop
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    avg_prop = (total_prop / len(loader.dataset)) if loss_fn == "cd_rep" else None
    return avg_loss, avg_prop


def main():
    # ----- logging setup -----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    log = logging.getLogger()

    log.info("Starting training script with Weights & Biases logging")
    log.info(f"Python executable: {sys.executable}")
    log.info(f"Torch version: {torch.__version__}")
    log.info(f"PyG version:   {torch_geometric.__version__}")

    # ----- load config -----
    conf = OmegaConf.load(os.path.join("conf", "config.yaml"))
    model_cfg  = conf.model_config
    train_cfg  = conf.train_config
    data_cfg   = conf.data_config

    log.info(f"Experiment name: {conf.name}")
    log.info("--- Config ---\n" + OmegaConf.to_yaml(conf).strip())

    # ----- init wandb -----
    wandb.init(
        project=conf.name,
        config=OmegaConf.to_container(conf, resolve=True),
        name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    # ----- dataset -----
    script_dir = os.getcwd()

    data_path = os.path.join(script_dir, 'data', 'custom', 'pu1k_patch_poisson_to_fps.h5')
    # data_path = os.path.join(script_dir, 'pu1k_fps256_uniform1024_from_meshes.h5')



    # Load the dataset
    dataset = PCDDataset.from_h5(
        data_path,
        num_point=256,        # this matches your input cloud size
        up_ratio=4,           # this matches 256 → 1024 (adjust if needed)
        skip_rate=1,          # or use data_config.skip_rate if defined
        seed=42               # or use data_config.rng_seed if defined
    )

    log.info(f"Dataset size: {len(dataset)} samples")
    loader = PyGLoader(dataset, batch_size=train_cfg.batch_size, follow_batch=["pos_s", "pos_t"])

    # ----- model -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    model = PUGCN(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, betas=train_cfg.betas)

    log.info("Model summary:")
    torchinfo.summary(model)

    # ----- prepare output dir -----
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    out_dir = os.path.join("trained-models", f"{ts}-{conf.name}")
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(config=conf, f=os.path.join(out_dir, "config.yaml"))
    log.info(f"Checkpoints & config will be saved to: {out_dir}")

    # ----- training loop -----
    start_time = time.time()
    for epoch in range(1, train_cfg.epochs + 1):
        epoch_start = time.time()
        loss, prop = train_one_epoch(
            model, loader, train_cfg.loss_fn, optimizer,
            gamma=(1 - epoch/train_cfg.epochs),
            device=device,
            k_loss=train_cfg.get("k_loss", 4)
        )
        epoch_time = time.time() - epoch_start
        msg = f"Epoch {epoch}/{train_cfg.epochs}  loss={loss:.6f}  time={epoch_time:.2f}s"
        if prop is not None:
            msg += f"  prop={prop:.4f}"
        log.info(msg)

        # Log metrics to wandb
        log_data = {"epoch": epoch, "loss": loss, "epoch_time": epoch_time}
        if prop is not None:
            log_data["prop"] = prop
        wandb.log(log_data)

        if epoch == 1 or epoch % train_cfg.save_every == 0:
            ckpt_path = os.path.join(out_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_size": model_size(model, unit="KB"),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            log.info(f"Saved checkpoint {ckpt_path}")
            wandb.save(ckpt_path)

    total_time = time.time() - start_time
    log.info(f"Training completed in {total_time:.2f} seconds")
    wandb.summary["total_time"] = total_time
    wandb.finish()

if __name__ == "__main__":
    main()
