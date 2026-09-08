"""RAD noise scheduler for RAD forward noising and reverse sampling."""
import torch

class RADNoiseScheduler:
    """Region_aware linear noise scheduler Phase1 : noise added only inside masked redion/ Phase2 : noise added outside mask"""

    def __init__(self, ph1_steps, ph2_steps, beta_start, beta_end,device):

        self.ph1_steps = ph1_steps
        self.ph2_steps = ph2_steps
        self.timpesteps= ph1_steps + ph2_steps
        self.device=device

        # phase 1 linear schedule
        self.betas1 = torch.linspace(beta_start, beta_end, ph1_steps,device=device)
        self.alphas1 = 1.0 - self.betas1
        self.alpha_bar1 = torch.cumprod(self.alphas1, dim=0)

        # phase 2 linear schedule
        self.betas2 = torch.linspace(beta_start, beta_end, ph2_steps,device=device)
        self.alphas2 = 1.0 - self.betas2
        self.alpha_bar2 = torch.cumprod(self.alphas2, dim=0)  

        #alpha_bar at previous local phase step
        self.alpha_bar_prev1 = torch.cat((torch.ones(1, device=device), self.alpha_bar1[:-1]))
        self.alpha_bar_prev2 = torch.cat((torch.ones(1, device=device), self.alpha_bar2[:-1]))

        
    def get_spatial_params(self, mask, t):
        """Build pixel-wise RAD diffusion parameters for timestep t."""

        B =mask.shape[0]
        shape = (B, 1, 1, 1)

        is_ph1 = t < self.ph1_steps

        t1 = torch.clamp(t, 0, self.ph1_steps - 1)
        t2 = torch.clamp(t - self.ph1_steps, 0, self.ph2_steps -1)

        beta1 = self.betas1[t1].reshape(shape)
        beta2 = self.betas2[t2].reshape(shape)

        alpha_bar1 = self.alpha_bar1[t1].reshape(shape)
        alpha_bar2 = self.alpha_bar2[t2].reshape(shape)

        alpha_bar_prev1 = self.alpha_bar_prev1[t1].reshape(shape)
        alpha_bar_prev2 = self.alpha_bar_prev2[t2].reshape(shape)

        is_ph1 = is_ph1.reshape(shape)

        b_t1 = mask * beta1
        b_t2 = (1.0 - mask) * beta2

        b_t = torch.where(is_ph1, b_t1, b_t2)
        a_t = 1.0 - b_t

        a_bar_t1 = mask * alpha_bar1 + (1.0 - mask)

        a_bar_t2 = (mask * self.alpha_bar1[-1] + (1.0 - mask) * alpha_bar2)

        a_bar_t = torch.where(is_ph1, a_bar_t1, a_bar_t2)

        b_bar_t = 1.0 - a_bar_t

        a_bar_prev1 = mask * alpha_bar_prev1 +(1.0 - mask)

        a_bar_prev2 = (mask * self.alpha_bar1[-1] + (1.0 - mask) * alpha_bar_prev2)

        a_bar_prev = torch.where(is_ph1, a_bar_prev1, a_bar_prev2)

        return b_t, a_t, a_bar_t, b_bar_t, a_bar_prev     
    
    def add_noise(self,x0,epsilon,mask,t):
        """Apply the RAD forward process at timestep t using the pixel-wise accumulated noise schedule."""

        _, _, a_bar_t, b_bar_t,_ = self.get_spatial_params(mask,t)
        xt = (torch.sqrt(a_bar_t)*x0 + torch.sqrt(b_bar_t)*epsilon)
        return xt, b_bar_t

    def reverse_sampling_step(self,xt, noise_pred, mask, t):
        """Perform one RAD reverse sampling step from x_t to x_{t-1} using pixel-wise diffusion parameters."""

        b_t, a_t, a_bar_t, b_bar_t, a_bar_prev = self.get_spatial_params(mask,t)
        active = b_t > 0

        denominator = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-12) )

        mu_active = (xt - (b_t * noise_pred) / denominator) / torch.sqrt(a_t)

        # positions that are not being diffused stay unchanged
        mu = torch.where(active,mu_active,xt)

        posterior_variance = ((1.0 - a_bar_prev)/ torch.clamp(1.0 - a_bar_t, min=1e-12)) * b_t

        posterior_variance = torch.where(active, posterior_variance, torch.zeros_like(posterior_variance))

        z = torch.randn_like(xt)
        xt_prev = (mu + torch.sqrt(torch.clamp(posterior_variance, min=0.0)) * z)
        x0_pred = (xt - torch.sqrt(b_bar_t) + noise_pred) / torch.sqrt(a_bar_t)

        return xt_prev, x0_pred