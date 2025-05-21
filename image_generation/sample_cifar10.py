import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from model.unet_arch import UNetModel
import argparse
import os


def find_model(model_name):
    """
    Load a custom checkpoint from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(
        model_name, map_location=lambda storage, loc: storage, weights_only=False)
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
    num_classes = args.num_classes if args.num_classes > 0 else None
    model = UNetModel(
        image_size=args.image_size,
        in_channels=3,
        out_channels=3,
        num_classes=num_classes
    ).to(device)
    # Load a custom model checkpoint from train.py:
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(
        prediction=args.prediction, 
        diffusion_type=args.diffusion_type,
        diffusion_steps=args.num_sampling_steps)

    n = 64 # default number of samples
    # Labels to condition the model with (feel free to change):
    if num_classes is not None:
        class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        y = torch.tensor(class_labels, device=device)
        n = len(class_labels)

    # Create sampling noise:
    z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
    
    # Setup classifier-free guidance:
    if num_classes is not None:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device) # 1000
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.forward_loop(
            model.forward_with_cfg, z, 
            num_steps=args.num_fast_sampling_steps, 
            sample_type=args.sample_type,
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=True, 
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    else:
        model_kwargs = dict()

        # Sample images:
        samples = diffusion.forward_loop(
            model, z, 
            num_steps=args.num_fast_sampling_steps, 
            sample_type=args.sample_type,
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=True, 
            device=device
        )

    # Save and display images:
    os.makedirs('result_samples', exist_ok=True)
    save_image(samples, "result_samples/sample.png", nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="U-Net")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128, 256, 512], default=32)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=4.0) # default 4
    parser.add_argument("--prediction", type=str, default="sflow")
    parser.add_argument("--diffusion-type", type=str, default="sde")
    parser.add_argument("--sample-type", type=str, default="MC")
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--num-fast-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, 
            default='pretrained/fod-sde.pt',
            help="Optional path to a model checkpoint.")
    args = parser.parse_args()
    main(args)
