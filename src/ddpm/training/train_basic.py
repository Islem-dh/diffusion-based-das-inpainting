"""Training utilities for basic DDPM noise-prediction model (simple unet)."""

import torch
from tqdm import tqdm

def train_ddpm_basic(model, train_loader,scheduler, optimizer, device, epochs, checkpoint_path, model_key, optimizer_key, loss_key, start_epoch=0, train_losses=None, checkpoint_meta=None, plot_function=None, loss_plot_path=None):
    """Train a DDPM model using MSE noise-prediction loss."""

    if train_losses is None : train_losses = []
    if checkpoint_meta is None : checkpoint_meta={}

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss= 0.0

        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):

            x=batch.to(device)
            epsilon=torch.randn_like(x)
            t=torch.randint(0,scheduler.betas.shape[0],(x.shape[0],),device=device).long()
            xt=scheduler.add_noise(x0=x, epsilon=epsilon,t=t)
            noise_prediction=model(xt,t)
        
            loss=((noise_prediction -epsilon)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()

        avg_loss=epoch_loss/len(train_loader)
        train_losses.append(avg_loss)
        torch.save(
            {
                model_key: model.state_dict(),
                optimizer_key: optimizer.state_dict(),
                loss_key: train_losses,
                "completed_epochs": epoch + 1,
                **checkpoint_meta,
            },
            checkpoint_path,
        )

        if plot_function is not None: plot_function(train_losses, loss_plot_path)

    if start_epoch >= epochs and plot_function is not None:
        plot_function(train_losses, loss_plot_path)    

    return train_losses    