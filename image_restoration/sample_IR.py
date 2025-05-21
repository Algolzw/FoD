import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusion import create_diffusion
from model.unet_arch import UNetModel
from data.LQGT_dataset import LQGTDataset
from torch.utils.data import DataLoader

from PIL import Image
import argparse
import os

from utils import tensor2img, calculate_psnr, calculate_ssim, calculate_lpips,validation
from data.util import bgr2ycbcr

import pyiqa
import time


def find_model(model_name):
    """
    Load a custom checkpoint from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.ckpt is not None

    # Load model:
    model = UNetModel(
        in_channels=6,
        out_channels=3,
        model_channels=64,
        channel_mult=[1, 2, 4, 8], 
    ).to(device)
    # Load a custom model checkpoint from train.py:
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(
        prediction=args.prediction,
        diffusion_type=args.diffusion_type,
        diffusion_steps=args.num_sampling_steps)

    
    # build dataset
    dataset = LQGTDataset(dataroot_LQ=args.LQ_path, dataroot_GT=args.HQ_path, phase='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    os.makedirs(args.sample_dir, exist_ok=True)

    # build eval metrics
    lpips_fn = pyiqa.create_metric('lpips', device=device)
    psnr, ssim, psnr_y, ssim_y, lpips = 0.0, 0.0, 0.0, 0.0, 0.0
    test_times = []

    for idx, data in enumerate(loader):
        gt_img = data['GT'].to(device)
        lq_img = data['LQ'].to(device)
        img_path = data['LQ_path'][0]
        img_name = os.path.basename(img_path)
        model_kwargs = dict()

        tic = time.time()
        # Sample images:
        hq_img = diffusion.forward_loop(
            model, 
            lq_img, 
            num_steps=args.num_fast_sampling_steps, 
            sample_type=args.sample_type,
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=True, device=device
        ).squeeze()
        toc = time.time()
        test_times.append(toc - tic)

        hq_img, gt_img = tensor2img(hq_img), tensor2img(gt_img)
        Image.fromarray(hq_img).save(f"{args.sample_dir}/{img_name}")

        psnr += calculate_psnr(hq_img, gt_img)
        ssim += calculate_ssim(hq_img, gt_img)
        lpips += calculate_lpips(lpips_fn, hq_img, gt_img, device)

        print(f"idx: {idx} - psnr: {psnr/len(test_times):.4f}, lpips: {lpips / len(test_times):.4f}")

        gt_y = bgr2ycbcr(gt_img, only_y=True)
        hq_y = bgr2ycbcr(hq_img, only_y=True)

        psnr_y += calculate_psnr(hq_y, gt_y)
        ssim_y += calculate_ssim(hq_y, gt_y)

        break


    n_samples = idx + 1
    avg_psnr = psnr / n_samples
    avg_ssim = ssim / n_samples
    avg_psnr_y = psnr_y / n_samples
    avg_ssim_y = ssim_y / n_samples
    avg_lpips = lpips / n_samples

    print(f"----Average PSNR/SSIM results for {args.task_name}: \
            PSNR: {avg_psnr:.6f} dB; SSIM: {avg_ssim:.6f}, LPIPS: {avg_lpips:.6f} \
            PSNR_Y: {avg_psnr_y:.6f} dB; SSIM_Y: {avg_ssim_y:.6f}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="fod")
    parser.add_argument("--task-name", type=str, default="deraining")

    parser.add_argument("--LQ-path", type=str, default='/home/ziwlu/datasets/rain/testH/LQ')
    parser.add_argument("--HQ-path", type=str, default='/home/ziwlu/datasets/rain/testH/GT')
    parser.add_argument("--sample-dir", type=str, default='samples')
    parser.add_argument("--model", type=str, default="U-Net")

    parser.add_argument("--prediction", type=str, default="sflow")
    parser.add_argument("--diffusion-type", type=str, default="sde")
    parser.add_argument("--sample-type", type=str, default="MC")
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--num-fast-sampling-steps", type=int, default=100)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='pretrained/fod-deraining.pt',
                        help="Optional path to a model checkpoint.")
    args = parser.parse_args()

    args.sample_dir = os.path.join(args.sample_dir, args.task_name, args.run_name)
    main(args)
