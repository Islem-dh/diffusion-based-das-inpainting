"""Unconditional DDPM prior with the Mamba U-Net, trained on 1001 (noise + earthquake).
Usage:  python scripts/train_prior_mamba.py   (resumes from the checkpoint if it exists)"""

import matplotlib
matplotlib.use("Agg")                                   
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from ddpm.data.das_dataset import DasDataset
from ddpm.diffusion.noise_scheduler import LinearNoiseScheduler
from ddpm.paths import CHECKPOINTS, DAS_DATA
from ddpm.models.unet_mamba import DDPM

das_data = DAS_DATA
train_files_earthquake = [das_data/"train_1001.h5"]
test_files_earthquake = [das_data/"test_1001.h5"]

device= "cuda" if torch.cuda.is_available() else "cpu"

train_set_earthquake = DasDataset(train_files_earthquake)
test_set_earthquake = DasDataset(test_files_earthquake)
train_loader_earthquake = DataLoader(train_set_earthquake, batch_size=16, shuffle=True)
test_loader_earthquake = DataLoader(test_set_earthquake, batch_size=16, shuffle=False)

model_earthquake_mamba=  DDPM(in_channels=1, out_channels=1, base_channels=64, channels_multiplier=(1,1,2,2,4),
                  nbr_rb=2, dropout=0.1).to(device)
scheduler_E = LinearNoiseScheduler(timesteps=1000, beta_start=1e-4,beta_end=0.02, device=device)

checkpoints=CHECKPOINTS
model_earthquake_path=checkpoints/"das_ddpm_earthquake_model_mamba.pth"
earthquake_loss_plot_path=checkpoints/"das_ddpm_earthquake_loss_mamba.png"

optimizer_earthquake_mamba= torch.optim.Adam(model_earthquake_mamba.parameters(), lr=1e-4)

epochs_earthquake= 40 
earthquake_train_losses=[]
earthquake_start_epoch=0


if model_earthquake_path.exists():
    checkpoint = torch.load(model_earthquake_path,map_location=device)
    if "model_earthquake_mamba" in checkpoint:
        model_earthquake_mamba.load_state_dict(checkpoint["model_earthquake_mamba"])
    if "optimizer_earthquake_mamba" in checkpoint:
        optimizer_earthquake_mamba.load_state_dict(checkpoint["optimizer_earthquake_mamba"])
    earthquake_train_losses= checkpoint.get("earthquake_train_losses", [])
    earthquake_start_epoch=checkpoint.get("completed_epochs", len(earthquake_train_losses)) 

for epoch in range(earthquake_start_epoch, epochs_earthquake):

    model_earthquake_mamba.train()
    epoch_loss= 0.0

    for batch in tqdm(train_loader_earthquake, desc=f"Epoch{epoch+1}/{epochs_earthquake}"):

        x=batch.to(device)
        epsilon=torch.randn_like(x)
        t=torch.randint(0,scheduler_E.betas.shape[0],(x.shape[0],),device=device).long()
        xt=scheduler_E.add_noise(x0=x, epsilon=epsilon,t=t)
        noise_prediction=model_earthquake_mamba(xt,t)
        
        loss=((noise_prediction -epsilon)**2).mean()

        optimizer_earthquake_mamba.zero_grad()
        loss.backward()
        optimizer_earthquake_mamba.step()
        epoch_loss+= loss.item()

    avg_loss=epoch_loss/len(train_loader_earthquake)
    earthquake_train_losses.append(avg_loss)
   
    torch.save({
    "model_earthquake_mamba": model_earthquake_mamba.state_dict(), "optimizer_earthquake_mamba": optimizer_earthquake_mamba.state_dict(),"earthquake_train_losses":earthquake_train_losses,
                  "completed_epochs": epoch+1, "base_channels": 64, "channels_multipilier": ( 1,1,2,2,4),
                  "normalizartion": {"function": "arcsinh", "s": 50.0, "A_max": 12350.0}
     }, model_earthquake_path)
    
    plt.figure(figsize = (6,4))
    plt.plot(range(1,len(earthquake_train_losses)+1), earthquake_train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Noise Prediction Loss")
    plt.title("DDPM Noise & Earthquake Loss")
    plt.savefig(earthquake_loss_plot_path)
    plt.close()
    print(f"epoch {epoch+1}/{epochs_earthquake} | train loss {avg_loss:.5f}", flush=True)

print("done:", model_earthquake_path)