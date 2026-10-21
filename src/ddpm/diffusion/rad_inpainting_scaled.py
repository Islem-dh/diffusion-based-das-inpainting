"""RAD inpainting with input scaling."""

from ddpm.diffusion.rad_inpainting import rad_inpaint


def inpaint_scaled(model,scheduler,x0,mask,device,scale=1.0,generator=None,timesteps_shown=None):
    """rad_inpaint with input scaling: model works on x0*scale, result is returned in normalized units."""

    res,x_start,steps=rad_inpaint(model=model,scheduler=scheduler,x0=x0*scale,mask=mask,device=device,
                                  generator=generator,timesteps_shown=timesteps_shown)

    steps={k:v/scale for k,v in steps.items()}

    return res/scale,x_start/scale,steps