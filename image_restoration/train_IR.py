import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import random

from model.unet_arch import UNetModel
from diffusion import create_diffusion
from data.LQGT_dataset import LQGTDataset

from utils import *

import pyiqa



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = set_seed(args.global_seed, rank)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    experiment_dir = f"{args.results_dir}/{args.run_name}-{args.prediction}"  # Create an experiment folder
    training_state_path = f"{experiment_dir}/training_state.pt"
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)


    # Create model:
    model = UNetModel(
        in_channels=6,
        out_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 4, 8], 
    )
    # Note that parameter initialization is done within the constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(
        prediction=args.prediction,
        diffusion_type=args.diffusion_type,
        diffusion_steps=args.num_sampling_steps)

    logger.info(f"Prediction: {args.prediction}, Diffusion steps: {args.num_sampling_steps}")
    logger.info(f"U-Net Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.niter, eta_min=1e-6)

    # Setup data:
    dataset = LQGTDataset(
        dataroot_LQ=args.LQ_path_train, dataroot_GT=args.HQ_path_train, size=args.image_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.LQ_path_train})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    epoch = 0
    start_time = time()

    if args.resume and os.path.exists(training_state_path):
        logger.info(f"Loading training state from {training_state_path}")
        epoch, train_steps = load_checkpoint(model, ema, opt, scheduler, rank, path=training_state_path)
        logger.info(f"epoch: {epoch}, train_steps: {train_steps}")

    # build evaluation metrics
    if rank <= 0:
        lpips_fn = pyiqa.create_metric('lpips', device=device)

    logger.info(f"Training for {args.niter} iternation...")
    while train_steps < args.niter:
        sampler.set_epoch(epoch)
        if epoch % 5 == 0:
            logger.info(f"Beginning epoch {epoch}...")

        for data in loader:
            y = data['GT'].to(device)
            x = data['LQ'].to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict()
            loss_dict = diffusion.training_losses(model, x, y, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"step={train_steps:07d}) Train Loss: {avg_loss:.6f}, Train Steps/Sec: {steps_per_sec:.2f}")
                ###################

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.val_every == 0 and train_steps > 0:
                if rank == 0:
                    avg_psnr, avg_lpips = validation(train_steps, model, diffusion, lpips_fn, args.LQ_path_val, args.HQ_path_val)
                    logger.info(f"<<<<<<<<<<step={train_steps:07d}, psnr={avg_psnr:.4f}, lpips={avg_lpips:.4f}>>>>>>>>>>")
                    save_checkpoint(model, ema, opt, scheduler, epoch, train_steps, training_state_path)
            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
        epoch += 1
        
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="fod")
    parser.add_argument("--task-name", type=str, default="deraining")

    parser.add_argument("--LQ-path-train", type=str, default='/home/ziwlu/datasets/rain/trainH/LQ')
    parser.add_argument("--HQ-path-train", type=str, default='/home/ziwlu/datasets/rain/trainH/GT')
    parser.add_argument("--LQ-path-val", type=str, default='/home/ziwlu/datasets/rain/testH/LQ')
    parser.add_argument("--HQ-path-val", type=str, default='/home/ziwlu/datasets/rain/testH/GT')
    parser.add_argument("--results-dir", type=str, default="results/rain")
    
    parser.add_argument("--prediction", type=str, default="sflow")
    parser.add_argument("--diffusion-type", type=str, default="sde")
    parser.add_argument('--resume', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num-sampling-steps", type=int, default=100)

    parser.add_argument("--model", type=str, default="U-Net")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--niter", type=int, default=500_000)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20_000)
    parser.add_argument("--val-every", type=int, default=2500)
    args = parser.parse_args()
    main(args)
