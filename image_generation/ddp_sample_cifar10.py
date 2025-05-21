"""
Samples a large number of images from a pre-trained model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample_cifar.py.
"""
import torch
import torch.distributed as dist
from model.unet_arch import UNetModel
from diffusion import create_diffusion
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse


def find_model(model_name):
    """
    Load a custom checkpoint from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    assert args.ckpt is not None

    # Load model:
    num_classes = args.num_classes if args.num_classes > 0 else None
    model = UNetModel(
        image_size=args.image_size,
        in_channels=3,
        out_channels=3,
        num_classes=num_classes
    ).to(device)
    # Load a custom DiT checkpoint from train.py:
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    # diffusion = create_diffusion(diffusion_steps=args.num_sampling_steps)
    diffusion = create_diffusion(
        prediction=args.prediction, 
        diffusion_type=args.diffusion_type,
        diffusion_steps=args.num_sampling_steps)

    if num_classes is not None:
        assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
        using_cfg = args.cfg_scale > 1.0
    else:
        using_cfg = False

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    # folder_name = f"{model_string_name}-{ckpt_string_name}-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    folder_name = f"{model_string_name}-{ckpt_string_name}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, args.image_size, args.image_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device) if num_classes is not None else None

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.forward_loop(
            sample_fn, z, 
            num_steps=args.num_fast_sampling_steps, 
            sample_type=args.sample_type,
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="U-Net")
    parser.add_argument("--sample-dir", type=str, default="samples/cifar10")
    parser.add_argument("--per-proc-batch-size", type=int, default=128)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128, 256, 512], default=32)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)

    parser.add_argument("--prediction", type=str, default="sflow")
    parser.add_argument("--diffusion-type", type=str, default="sde")
    parser.add_argument("--sample-type", type=str, default="MC")
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--num-fast-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default='pretrained/fod-sde.pt',
                        help="Optional path to a checkpoint.")
    args = parser.parse_args()
    main(args)
