# train_dit_hf.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP, adapted to use
the Hugging Face "timm/imagenet-1k-wds" webdataset-backed ImageNet.
"""

import os
import argparse
import logging
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms

from datasets import load_dataset
from diffusers.models import AutoencoderKL

from models import DiT_models
from diffusion import create_diffusion


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0 and logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(),
                      logging.FileHandler(f"{logging_dir}/log.txt")]
        )
    logger = logging.getLogger(__name__)
    if dist.get_rank() != 0:
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size,
                           crop_x:crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Requires at least one GPU."

    # Setup DDP:
    # dist.init_process_group("nccl")
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    assert args.global_batch_size % world_size == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder (only on rank 0):
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model + EMA:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size,
                                   num_classes=args.num_classes)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    # Diffusion + VAE:
    diffusion = create_diffusion(timestep_respacing="")  # 1000 steps linear
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # -------------------------------------------------------------------------
    # Setup data using Hugging Face timm/imagenet-1k-wds
    # -------------------------------------------------------------------------
    ds = load_dataset("timm/imagenet-1k-wds", cache_dir="data/imagenet")
    train_hf = ds["train"]

    # define transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True)
    ])

    # wrap HF dataset
    class HFDataset(Dataset):
        def __init__(self, hf_dataset, transform):
            self.hf = hf_dataset
            self.tf = transform

        def __len__(self):
            return len(self.hf)

        def __getitem__(self, idx):
            ex = self.hf[idx]
            img = ex["jpg"]
            # if loaded as numpy array
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            x = self.tf(img)
            y = ex["json"]["label"]
            return x, y

    dataset = HFDataset(train_hf, transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // world_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} training examples (HuggingFace)")

    # Prepare models for training
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Training loop
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            # print(x.shape, y.shape)
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps,
                              (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, {"y": y})
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps,
                                        device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) "
                            f"Train Loss: {avg_loss:.4f}, "
                            f"Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    ckpt = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(ckpt, path)
                    logger.info(f"Saved checkpoint to {path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=False,
                        help="(unused when using HF dataset)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()),
                        default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512],
                        default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"],
                        default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
