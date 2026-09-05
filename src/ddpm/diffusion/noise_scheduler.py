"""Linear noise scheduler for DDPM forward noising and reverse sampling."""
import torch

class LinearNoiseScheduler:
    """Implements a linear beta schedule for DDPM training and sampling."""
    def __init__(self,timesteps,beta_start, beta_end,device):
        self.timpesteps=timesteps
        self.device=device

        # Linear beta schedule over all diffusion timesteps.
        self.betas=torch.linspace(beta_start, beta_end, timesteps,device=device)
        self.alphas= 1.0-self.betas
        self.alpha_bar=torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar=torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar=torch.sqrt(1.0-self.alpha_bar)

    def add_noise(self,x0,epsilon,t):
        """Add Gaussian noise to clean samples x0 at diffusion timestep t."""
        sqrt_alpha_bar_t= self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar_t=self.sqrt_one_minus_alpha_bar[t]
        if t.ndim > 0 :
            batch_size= x0.shape[0]
            shape= (batch_size,) + (1,)*(x0.ndim -1)
            sqrt_alpha_bar_t= sqrt_alpha_bar_t.reshape(shape)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.reshape(shape)
        
        xt=sqrt_alpha_bar_t*x0 + sqrt_one_minus_alpha_bar_t * epsilon
        return xt

    def reverse_sampling_step(self,xt, noise_pred, t):
        """Perform one DDPM reverse sampling step from x_t to x_{t-1}."""
        x0_pred= (xt - self.sqrt_one_minus_alpha_bar[t] * noise_pred) / self.sqrt_alpha_bar[t]
        mu_nom= xt-(self.betas[t]*noise_pred)/self.sqrt_one_minus_alpha_bar[t]
        mu= mu_nom/torch.sqrt(self.alphas[t])

        if t>0:

            std= torch.sqrt(((1.0- self.alpha_bar[t-1])/(1.0 -self.alpha_bar[t]))*self.betas[t])
            z=torch.randn_like(xt)
            xt_prev = mu +std*z
        else : xt_prev=mu

        return xt_prev, x0_pred