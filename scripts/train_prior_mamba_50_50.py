"""Unconditional DDPM prior with the Mamba U-Net, trained on a 50/50 mix:
50 % noise-only patches (1000) and 50 % noise + earthquake patches (1001), standard MSE.
Usage:  python scripts/train_prior_mamba_50_50.py   (resumes from the checkpoint if it exists)"""

import matplotlib
matplotlib.use("Agg")                                   
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm
from ddpm.data.das_dataset import DasDataset
from ddpm.diffusion.noise_scheduler import LinearNoiseScheduler
from ddpm.models.unet_mamba import DDPM
from ddpm.paths import CHECKPOINTS, DAS_DATA

das_data = DAS_DATA
train_set_1001 = DasDataset([das_data/"train_1001.h5"])
train_set_1000 = DasDataset([das_data/"train_1000.h5"])

test_set_1001 = DasDataset([das_data/"test_1001.h5"])
test_set_1000 = DasDataset([das_data/"test_1000.h5"])

device = "cuda" if torch.cuda.is_available() else "cpu"

total_train_patches = 16384
e90 = int(total_train_patches * 0.50)
n10 = total_train_patches - e90

total_val_patches = 512
e90v = int(total_val_patches * 0.50)
n10v = total_val_patches - e90v

seed = torch.Generator().manual_seed(42)
i_n = torch.randperm(len(train_set_1000), generator=seed)[:n10]
i_e = torch.randperm(len(train_set_1001), generator=seed)[:e90]
subset_noise = Subset(train_set_1000, i_n)
subset_earthquake = Subset(train_set_1001, i_e)
train_set_mix = ConcatDataset([subset_noise, subset_earthquake])

seed_v = torch.Generator().manual_seed(42)
i_nv = torch.randperm(len(test_set_1000), generator=seed_v)[:n10v]
i_ev = torch.randperm(len(test_set_1001), generator=seed_v)[:e90v]
subset_noise_v = Subset(test_set_1000, i_nv)
subset_earthquake_v = Subset(test_set_1001, i_ev)
test_set_mix= ConcatDataset([subset_noise_v, subset_earthquake_v])

train_loader_mix = DataLoader(train_set_mix, batch_size=16 , shuffle=True)
print("total:", len(train_set_mix), "N+E:", len(subset_earthquake), "N:", len(subset_noise), "batches:" , len(train_loader_mix))
test_loader_mix = DataLoader(test_set_mix, batch_size=16 , shuffle=False)
print("total:", len(test_set_mix), "N+E:", len(subset_earthquake_v), "N:", len(subset_noise_v), "batches:" , len(test_loader_mix))

model_mb=  DDPM(in_channels=1, out_channels=1, base_channels=64, channels_multiplier=(1,1,2,2,4),
                  nbr_rb=2, dropout=0.1).to(device)
scheduler_E = LinearNoiseScheduler(timesteps=1000, beta_start=1e-4,beta_end=0.02, device=device)

checkpoints=CHECKPOINTS
model_mb_path=checkpoints/"das_ddpm_earthquake_model_mamba_50_50_mixed_normal_mse_c.pth"
model_mb_loss_plot_path=checkpoints/"das_ddpm_earthquake_loss_mamba_50_50_mixed_normal_mse_c.png"

optimizer_model_mb= torch.optim.Adam(model_mb.parameters(), lr=1e-4)

epochs_earthquake= 20 
earthquake_train_losses=[]
earthquake_val_losses = []
earthquake_start_epoch=0


if model_mb_path.exists():
    checkpoint = torch.load(model_mb_path,map_location=device)
    if "model_mb" in checkpoint:
        model_mb.load_state_dict(checkpoint["model_mb"])
    if "optimizer_model_mb" in checkpoint:
        optimizer_model_mb.load_state_dict(checkpoint["optimizer_model_mb"])
    earthquake_train_losses= checkpoint.get("earthquake_train_losses", [])
    earthquake_val_losses= checkpoint.get("earthquake_val_losses", [])
    earthquake_start_epoch=checkpoint.get("completed_epochs", len(earthquake_train_losses)) 

for epoch in range(earthquake_start_epoch, epochs_earthquake):

    model_mb.train()
    epoch_loss= 0.0

    for batch in tqdm(train_loader_mix, desc=f"Epoch{epoch+1}/{epochs_earthquake}"):

        x=batch.to(device)
        epsilon=torch.randn_like(x)
        t=torch.randint(0,scheduler_E.betas.shape[0],(x.shape[0],),device=device).long()
        xt=scheduler_E.add_noise(x0=x, epsilon=epsilon,t=t)
        noise_prediction=model_mb(xt,t)
        
        loss = ((noise_prediction - epsilon)**2).mean()

        optimizer_model_mb.zero_grad()
        loss.backward()
        optimizer_model_mb.step()
        epoch_loss+= loss.item()

    avg_train_loss=epoch_loss/len(train_loader_mix)
    earthquake_train_losses.append(avg_train_loss)

    #validation
    model_mb.eval()
    val_epoch_loss = 0.0
    val_seed = torch.Generator(device=device).manual_seed(42)

    with torch.no_grad():
        for batch in test_loader_mix:
            x_v = batch.to(device)
            epsilon_v = torch.randn(x_v.shape, device=device, generator=val_seed)
            t_v = torch.randint(0,scheduler_E.betas.shape[0],(x_v.shape[0],),device=device, generator=val_seed).long()
            xt_v=scheduler_E.add_noise(x0=x_v, epsilon=epsilon_v,t=t_v)
            noise_prediction_v = model_mb(xt_v,t_v)

            loss_v=((noise_prediction_v - epsilon_v)**2).mean()
            val_epoch_loss += loss_v.item()
            

    avg_val_loss = (val_epoch_loss / len(test_loader_mix))
    earthquake_val_losses.append(avg_val_loss)

    torch.save({
    "model_mb": model_mb.state_dict(), "optimizer_model_mb": optimizer_model_mb.state_dict(),"mse": "normal mse","earthquake_train_losses":earthquake_train_losses,
    "earthquake_val_losses": earthquake_val_losses,"loader": "mixed_50_50", "completed_epochs": epoch+1, "base_channels": 64, "channels_multipilier": ( 1,1,2,2,4),
                  "normalizartion": {"function": "arcsinh", "s": 50.0, "A_max": 12350.0}
     }, model_mb_path)
    
    plt.figure(figsize = (6,4))
    plt.plot(range(1,len(earthquake_train_losses)+1), earthquake_train_losses, marker="o", label = "Training Loss")
    plt.plot(range(1,len(earthquake_val_losses)+1), earthquake_val_losses, marker="o", label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Noise Prediction Loss")
    plt.title("DDPM Mamba 50/50 mixed Prior - Standard MSE")
    plt.legend()
    plt.savefig(model_mb_loss_plot_path)
    plt.close()
    print(f"epoch {epoch+1}/{epochs_earthquake} | train {avg_train_loss:.5f} | val {avg_val_loss:.5f}", flush=True)

print("done:", model_mb_path)