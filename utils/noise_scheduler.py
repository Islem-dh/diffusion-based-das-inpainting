import torch

class LinearNoiseScheduler:
    def __init__(self,timesteps,beta_start, beta_end,device):
        self.timpesteps=timesteps
        self.device=device
        #self.beta_start=beta_start
        #self.beta_end=beta_end

        self.betas=torch.linspace(beta_start, beta_end, timesteps,device=device)
        self.alphas= 1.0-self.betas
        self.alpha_bar=torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar=torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar=torch.sqrt(1.0-self.alpha_bar)

    def add_noise(self,x0,epsilon,t):
        #t= np.asarray(t)
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
        x0_pred= (xt - self.sqrt_one_minus_alpha_bar[t] * noise_pred) / self.sqrt_alpha_bar[t]
        #for mnist in [0..1]
        #x0_pred =torch.clamp(x0_pred, 0.0, 1.0)
        mu_nom= xt-(self.betas[t]*noise_pred)/self.sqrt_one_minus_alpha_bar[t]
        mu= mu_nom/torch.sqrt(self.alphas[t])

        if t>0:

            std= torch.sqrt(((1.0- self.alpha_bar[t-1])/(1.0 -self.alpha_bar[t]))*self.betas[t])
            #z= np.random.randn()
            z=torch.randn_like(xt)
            xt_prev = mu +std*z
        else : xt_prev=mu

        return xt_prev, x0_pred