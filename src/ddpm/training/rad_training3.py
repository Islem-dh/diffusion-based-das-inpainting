"""Training for RAD Model"""

import copy
import torch
from tqdm import tqdm
from ddpm.masks.rad_masks import random_rad_mask


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """EMA of the model weights: ema = decay * ema + (1 - decay) * model"""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def sample_timesteps(B, scheduler, device, phase1_prob=None, generator=None):
    """Uniform timesteps, or phase 1 sampled with probability phase1_prob."""
    if phase1_prob is None:
        return torch.randint(0, scheduler.timesteps, (B,), device=device, generator=generator).long()
    is_ph1 = torch.rand(B, device=device, generator=generator) < phase1_prob
    t1 = torch.randint(0, scheduler.ph1_steps, (B,), device=device, generator=generator)
    t2 = torch.randint(scheduler.ph1_steps, scheduler.timesteps, (B,), device=device, generator=generator)
    return torch.where(is_ph1, t1, t2).long()


def masked_mse(noise_prediction, epsilon, b_bar_t):
    """MSE only on pixels that are actually noised (b_bar_t > 0)."""
    w = (b_bar_t > 0).float()
    return (((noise_prediction - epsilon) ** 2) * w).sum() / w.sum().clamp(min=1.0)


def train_rad(model, train_loader, val_loader, scheduler, optimizer, device, epochs, checkpoint_path,
              model_key, optimizer_key, train_loss_key, val_loss_key, start_epoch=0, 
              train_losses=None, val_losses=None, checkpoint_meta=None, plot_function=None, loss_plot_path=None,
              ema_model=None, ema_decay=0.999, phase1_prob=0.75, masked_loss=True, data_scale=1.0):
    
    if train_losses is None : train_losses = []
    if val_losses is None : val_losses = []
    if checkpoint_meta is None : checkpoint_meta={}

    # EMA copy of the model (created from the current weights if none is given)
    if ema_model is None:
        ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters(): p.requires_grad_(False)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_train_loss= 0.0
    
        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
    
            # input scaling: normalized data * data_scale (global train std -> 0.5)
            x=batch.to(device) * data_scale
            B,_,H,W = x.shape
            epsilon=torch.randn_like(x)
            t=sample_timesteps(B, scheduler, device, phase1_prob)
            mask = random_rad_mask(B,H,W,device)

            xt, b_bar_t=scheduler.add_noise(x0=x, epsilon=epsilon,mask=mask,t=t)
            noise_prediction=model(xt,b_bar_t)
            
            if masked_loss: loss=masked_mse(noise_prediction, epsilon, b_bar_t)
            else: loss=((noise_prediction -epsilon)**2).mean()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model, ema_decay)
            epoch_train_loss+= loss.item()
    
        avg_train_loss=epoch_train_loss/len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation (on the EMA model, same loss and same scaling as training)
        epoch_val_loss=0.0
        val_seed = torch.Generator(device=device).manual_seed(42)
        with torch.no_grad():
            for batch in val_loader:
                x_v = batch.to(device) * data_scale
                Bv,_,Hv,Wv = x_v.shape
                mask_v = random_rad_mask(Bv,Hv,Wv,device, generator=val_seed)
                epsilon_v = torch.randn(x_v.shape, device=device, generator=val_seed)
                t_v = sample_timesteps(Bv, scheduler, device, phase1_prob, generator=val_seed)
                xt_v, b_bar_t_v=scheduler.add_noise(x0=x_v, epsilon=epsilon_v, mask=mask_v, t=t_v)
                noise_prediction_v = ema_model(xt_v, b_bar_t_v )
                if masked_loss: loss_v = masked_mse(noise_prediction_v, epsilon_v, b_bar_t_v)
                else: loss_v = ((noise_prediction_v -epsilon_v)**2).mean()
                epoch_val_loss+= loss_v.item()
        avg_val_loss = epoch_val_loss/len(val_loader)
        val_losses.append(avg_val_loss)  

        torch.save(
            {
                model_key: model.state_dict(),
                "ema_" + model_key: ema_model.state_dict(),
                optimizer_key: optimizer.state_dict(),
                train_loss_key: train_losses,
                val_loss_key: val_losses,
                "completed_epochs": epoch + 1,
                "ema_decay": ema_decay, "phase1_prob": phase1_prob, "masked_loss": masked_loss,
                "data_scale": data_scale,
                **checkpoint_meta,
            },
            checkpoint_path,
        )
    
        if plot_function is not None: plot_function(train_losses,val_losses, loss_plot_path)
    
    if start_epoch >= epochs and plot_function is not None:
        plot_function(train_losses, val_losses, loss_plot_path)    
    
    return train_losses, val_losses, ema_model