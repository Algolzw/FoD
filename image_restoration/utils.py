import math
import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from data.LQGT_dataset import LQGTDataset

from collections import OrderedDict
from tqdm import tqdm

import logging
import pyiqa


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def set_seed(seed, rank):
    seed = seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def save_checkpoint(model, ema, optimizer, scheduler, epoch, train_steps, path='training_state.pt'):
    torch.save({
        'epoch': epoch,
        'train_steps': train_steps,
        'model_state_dict': model.module.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)

def load_checkpoint(model, ema, optimizer, scheduler, rank, path='training_state.pt'):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.module.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'] + 1, checkpoint['train_steps']

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
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
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


##########
##########

def img2tensor(img, normalize=False):
    """
    # RGB, HWC to CHW, numpy to tensor
    Input: img(H, W, C), [0,255], np.uint8 (default)
    Output: 3D(C,H,W), RGB order, float tensor
    """
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    if normalize:
        img = img * 2 - 1
    return img

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")

def calculate_lpips(lpips_fn, img1, img2, device):
    # img1 and img2 have range [0, 255]
    img1 = img2tensor(img1, False).unsqueeze(0).to(device)
    img2 = img2tensor(img2, False).unsqueeze(0).to(device)
    lp_score = lpips_fn(img1, img2).squeeze().item()
    return lp_score


def validation(step, model, diffusion, lpips_fn, LQ_path, HQ_path, sample_dir="images/result_samples"):
    model.eval()
    device = next(model.parameters()).device
    # build datasets
    dataset = LQGTDataset(dataroot_LQ=LQ_path, dataroot_GT=HQ_path, phase='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    os.makedirs(sample_dir, exist_ok=True)

    avg_psnr, avg_lpips = 0.0, 0.0
    for idx, data in enumerate(loader):
        y = data['GT'].to(device)
        x = data['LQ'].to(device)
        img_path = data['LQ_path'][0]
        img_name = os.path.basename(img_path)
        model_kwargs = dict()

        # Sample images:
        sample = diffusion.forward_loop(
            model, x, 
            num_steps=-1, 
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=True, device=device
        ).squeeze()

        sample, y = tensor2img(sample), tensor2img(y)
        Image.fromarray(sample).save(f"{sample_dir}/{img_name}")

        avg_psnr += calculate_psnr(sample, y)
        avg_lpips += calculate_lpips(lpips_fn, sample, y, device)

        if idx >= 10:
            break

    n_samples = idx + 1
    avg_psnr = avg_psnr / n_samples
    avg_lpips = avg_lpips / n_samples

    model.train()
    return avg_psnr, avg_lpips












