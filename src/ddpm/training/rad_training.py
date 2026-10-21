"""Training for RAD Model"""

import torch
from tqdm import tqdm
from ddpm.masks.rad_masks import random_rad_mask

def train_rad(model, train_loader, val_loader, scheduler, optimizer, device, epochs, checkpoint_path,
              model_key, optimizer_key, train_loss_key, val_loss_key, start_epoch=0, 
              train_losses=None, val_losses=None, checkpoint_meta=None, plot_function=None, loss_plot_path=None):
    
    if train_losses is None : train_losses = []
    if val_losses is None : val_losses = []
    if checkpoint_meta is None : checkpoint_meta={}
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_train_loss= 0.0
    
        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
    
            x=batch.to(device)
            B,_,H,W = x.shape
            epsilon=torch.randn_like(x)
            t=torch.randint(0,scheduler.timesteps,(x.shape[0],),device=device).long()
            mask = random_rad_mask(B,H,W,device)

            xt, b_bar_t=scheduler.add_noise(x0=x, epsilon=epsilon,mask=mask,t=t)
            noise_prediction=model(xt,b_bar_t)
            
            loss=((noise_prediction -epsilon)**2).mean()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss+= loss.item()
    
        avg_train_loss=epoch_train_loss/len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation
        model.eval()
        epoch_val_loss=0.0
        val_seed = torch.Generator(device=device).manual_seed(42)
        with torch.no_grad():
            for batch in val_loader:
                x_v = batch.to(device)
                Bv,_,Hv,Wv = x_v.shape
                mask_v = random_rad_mask(Bv,Hv,Wv,device, generator=val_seed)
                epsilon_v = torch.randn(x_v.shape, device=device, generator=val_seed)
                t_v = torch.randint(0,scheduler.timesteps,(x_v.shape[0],),device=device, generator=val_seed).long()
                xt_v, b_bar_t_v=scheduler.add_noise(x0=x_v, epsilon=epsilon_v, mask=mask_v, t=t_v)
                noise_prediction_v = model(xt_v, b_bar_t_v )
                loss_v = ((noise_prediction_v -epsilon_v)**2).mean()
                epoch_val_loss+= loss_v.item()
        avg_val_loss = epoch_val_loss/len(val_loader)
        val_losses.append(avg_val_loss)  

        torch.save(
            {
                model_key: model.state_dict(),
                optimizer_key: optimizer.state_dict(),
                train_loss_key: train_losses,
                val_loss_key: val_losses,
                "completed_epochs": epoch + 1,
                **checkpoint_meta,
            },
            checkpoint_path,
        )
    
        if plot_function is not None: plot_function(train_losses,val_losses, loss_plot_path)
    
    if start_epoch >= epochs and plot_function is not None:
        plot_function(train_losses, val_losses, loss_plot_path)    
    
    return train_losses, val_losses