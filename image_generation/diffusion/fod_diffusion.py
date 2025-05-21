import math
import numpy as np

import torch
import torchvision.utils as tvutils
import enum


def get_jsd_schedule(num_diffusion_timesteps, scale=1.5):
    betas = 1. / np.linspace(
        num_diffusion_timesteps, 1., num_diffusion_timesteps, dtype=np.float64
    )
    return betas ** scale

def get_linear_schedule(num_diffusion_timesteps):
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

def get_cosine_schedule(num_diffusion_timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = num_diffusion_timesteps + 1
    t = np.linspace(0, num_diffusion_timesteps, steps) / num_diffusion_timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return betas_clipped

def get_named_schedule(schedule_name, timesteps):
    if schedule_name == 'jsd':
        schedule = get_jsd_schedule(timesteps)
    elif schedule_name == 'linear':
        schedule = get_linear_schedule(timesteps)
    elif schedule_name == 'cosine':
        schedule = get_cosine_schedule(timesteps)
    elif schedule_name == 'const':
        schedule = np.ones(timesteps)
    elif schedule_name == 'none':
        schedule = np.zeros(timesteps)
    else:
        print('Not implemented such schedule for sigmas yet!!!')

    return schedule


#############################################################################
#############################################################################
#############################################################################

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    FINAL_X = enum.auto()  # the model predicts x_T
    FLOW = enum.auto()  # the model predicts x_T - x_0
    SFLOW = enum.auto()  # the model predicts x_T - x_t

def to_tensor(nparray):
    return torch.from_numpy(nparray).float()


class FoDiffusion:
    def __init__(self, thetas, sigmas, model_type, sigmas_scale=1.):
        super().__init__()
        self.model_type = model_type

        thetas = np.array(thetas, dtype=np.float64)
        sigmas = np.array(sigmas, dtype=np.float64)
        self.thetas = np.append(0.0, thetas)
        if np.sum(sigmas) > 0:
            sigmas = sigmas_scale * sigmas / np.sum(sigmas) # normalize sigmas
        self.sigmas = np.append(0.0, sigmas)
        expo_mean = -(self.thetas + 0.5 * self.sigmas)

        self.thetas_cumsum = np.cumsum(self.thetas)
        self.sigmas_cumsum = np.cumsum(self.sigmas)
        expo_mean_cumsum = -(self.thetas_cumsum + 0.5 * self.sigmas_cumsum)

        self.dt = math.log(0.001) / expo_mean_cumsum[-1]

        #### sqrt terms  ####
        self.expo_mean = expo_mean * self.dt
        self.sqrt_expo_variance = np.sqrt(self.sigmas * self.dt)
        self.expo_mean_cumsum = expo_mean_cumsum * self.dt
        self.sqrt_expo_variance_cumsum = np.sqrt(self.sigmas_cumsum * self.dt)

        self.num_timesteps = int(thetas.shape[0])

    #####################################

    def expo_normal(self, t, noise=None):
        assert noise is not None
        return torch.exp(
            _extract_into_tensor(self.expo_mean, t, noise.shape) + 
            _extract_into_tensor(self.sqrt_expo_variance, t, noise.shape) * noise
        )

    def expo_normal_cumsum(self, t, noise=None):
        assert noise is not None
        return torch.exp(
            _extract_into_tensor(self.expo_mean_cumsum, t, noise.shape) + 
            _extract_into_tensor(self.sqrt_expo_variance_cumsum, t, noise.shape) * noise
        )

    def expo_normal_transition(self, s, t, noise=None):
        assert noise is not None
        expo_mean_cumsum = _extract_into_tensor(self.expo_mean_cumsum, t, noise.shape) \
                            - _extract_into_tensor(self.expo_mean_cumsum, s, noise.shape)
        expo_variance_cumsum = _extract_into_tensor(self.sigmas_cumsum * self.dt, t, noise.shape) \
                            - _extract_into_tensor(self.sigmas_cumsum * self.dt, s, noise.shape)

        return torch.exp(expo_mean_cumsum + torch.sqrt(expo_variance_cumsum) * noise)


    def sde_step(self, x, x_final, t, noise):
        drift = _extract_into_tensor(self.thetas, t, x.shape) * (x_final - x)
        diffusion = _extract_into_tensor(self.sigmas, t, x.shape) * (x - x_final)
        return x + drift * self.dt + diffusion * math.sqrt(self.dt) * noise

    def forward_step(
        self, 
        model, 
        x, x_start, 
        t, t_next, 
        sample_type="EM", 
        clip_denoised=True, model_kwargs=None):

        model_output = model(x, t, **model_kwargs)

        if self.model_type == ModelType.FINAL_X:
            x_final = model_output
        elif self.model_type == ModelType.FLOW:
            x_final = x_start + model_output
        elif self.model_type == ModelType.SFLOW:
            x_final = x + model_output

        if clip_denoised:
            x_final = x_final.clamp(-1, 1)
        
        noise = torch.randn_like(x)
        if sample_type == "EM":
            x = self.sde_step(x, x_final, t, noise)
        elif sample_type == "MC":
            x = (x - x_final) * self.expo_normal_transition(t, t_next, noise) + x_final
        elif sample_type == "NMC":
            x = (x - x_final) * self.expo_normal_cumsum(t, t_next, noise) + x_final
        return x

    # forward process to get x(T) from x(0)
    def forward_loop(
        self, 
        model, 
        x_start, 
        num_steps=-1,
        sample_type="EM", # EM, MC, NMC
        clip_denoised=True, 
        model_kwargs=None, 
        device=None, 
        progress=False):

        assert x_start is not None
        if device is None:
            device = next(model.parameters()).device

        if num_steps <= 0:
            num_steps = self.num_timesteps

        img = x_start
        indices = np.linspace(0, self.num_timesteps, num_steps + 1).astype(int)
        times = np.copy(indices)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(enumerate(indices[:-1]))
        else:
            indices = enumerate(indices[:-1])

        for i, idx in indices:
            t = torch.tensor([idx] * x_start.shape[0], device=device)
            t_next = torch.tensor([times[i+1]] * x_start.shape[0], device=device)
            with torch.no_grad():
                img = self.forward_step(
                    model, img, x_start, t, t_next, sample_type, 
                    clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        return img

    ################################################################

    # sample states for training
    def training_losses(self, model, x_final, t, model_kwargs=None, noise=None):

        if model_kwargs is None:
            model_kwargs = {}

        x_start = torch.randn_like(x_final)
        if noise is None:
            noise = torch.randn_like(x_final)

        # generate states
        x_t = (x_start - x_final) * self.expo_normal_cumsum(t, noise) + x_final

        # model prediction
        model_output = model(x_t, t, **model_kwargs)

        target = {
            ModelType.FINAL_X: x_final,
            ModelType.FLOW: x_final - x_start,
            ModelType.SFLOW: x_final - x_t,
        }[self.model_type]

        terms = {}
        terms["loss"] = mean_flat((target - model_output)**2)

        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


