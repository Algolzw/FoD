from . import fod_diffusion as sd


def create_diffusion(
    theta_schedule="cosine",
    sigma_schedule="linear",
    diffusion_type="sde",
    prediction="sflow", # default sflow
    diffusion_steps=100,
):
    
    if diffusion_type == "ode":
        sigma_schedule = "none"
        
    thetas = sd.get_named_schedule(theta_schedule, diffusion_steps)
    sigma2s = sd.get_named_schedule(sigma_schedule, diffusion_steps)

    if prediction=="final":
        model_type = sd.ModelType.FINAL_X
    elif prediction=="flow":
        model_type = sd.ModelType.FLOW
    elif prediction=="sflow":
        model_type = sd.ModelType.SFLOW
    else:
        print("Prediction only supports: final, flow, nflow!\n Use flow by defualt!")
        model_type = sd.ModelType.SFLOW

    return sd.FoDiffusion(
        thetas=thetas,
        sigma2s=sigma2s,
        model_type=model_type,
    )
