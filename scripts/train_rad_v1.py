"""RAD v1: Region-Aware Diffusion, 40 epochs, plain MSE on all pixels, uniform timesteps, no EMA.
Usage:  python scripts/train_rad_v1.py   (resumes from the checkpoint if it exists)"""

import matplotlib
matplotlib.use("Agg")                                   
import copy
import json
import torch
from torch.utils.data import DataLoader
from ddpm.data.das_dataset import DasDataset
from ddpm.diffusion.rad_noise_scheduler import RADNoiseScheduler
from ddpm.models.unet_rad import RAD
from ddpm.plotting.loss_plot import update_loss_plot
from ddpm.paths import CHECKPOINTS, DAS_DATA
from ddpm.training.rad_training import train_rad

das_data = DAS_DATA
train_files_earthquake = [das_data/"train_1001.h5"]
test_files_earthquake = [das_data/"test_1001.h5"]

device= "cuda" if torch.cuda.is_available() else "cpu"

train_set_earthquake = DasDataset(train_files_earthquake)
test_set_earthquake = DasDataset(test_files_earthquake)
train_loader_earthquake = DataLoader(train_set_earthquake, batch_size=16, shuffle=True)
test_loader_earthquake = DataLoader(test_set_earthquake, batch_size=16, shuffle=False)

model_rad =  RAD(in_channels=1, out_channels=1, base_channels=64, channels_multiplier=(1,1,2,2,4),
                  nbr_rb=2, dropout=0.1).to(device)

scheduler_rad = RADNoiseScheduler(ph1_steps=1000, ph2_steps=1000, beta_start=1e-4,beta_end=0.02, device=device)

checkpoints=CHECKPOINTS
model_path=checkpoints/"das_rad_earthquake_model.pth"
loss_plot_path=checkpoints/"das_rad_earthquake_loss.png"

optimizer_rad= torch.optim.Adam(model_rad.parameters(), lr=1e-4)

epochs= 40 
train_losses=[]
val_losses=[]
start_epoch=0

checkpoint_meta = {
    "base_channels": 64, "channels_multiplier": ( 1,1,2,2,4),
                  "normalization": {"function": "arcsinh", "s": 50.0, "A_max": 12350.0},
    "model_key" : "model_rad", "optimizer_key": "optimizer_rad", "train_loss_key": "train_losses",
    "val_loss_key": "val_losses"
}

if model_path.exists():
    checkpoint = torch.load(model_path,map_location=device)
    if "model_rad" in checkpoint:
        model_rad.load_state_dict(checkpoint["model_rad"])
    if "optimizer_rad" in checkpoint:
        optimizer_rad.load_state_dict(checkpoint["optimizer_rad"])
    train_losses= checkpoint.get("train_losses", [])
    val_losses= checkpoint.get("val_losses", [])
    start_epoch=checkpoint.get("completed_epochs", len(train_losses)) 

train_losses, val_losses = train_rad(
    model=model_rad,
    train_loader=train_loader_earthquake,
    val_loader=test_loader_earthquake,
    scheduler=scheduler_rad,
    optimizer=optimizer_rad,
    device=device,
    epochs=epochs,
    checkpoint_path=model_path,
    model_key="model_rad",
    optimizer_key="optimizer_rad",
    train_loss_key="train_losses",
    val_loss_key="val_losses",
    start_epoch=start_epoch,
    train_losses=train_losses,
    val_losses=val_losses,
    checkpoint_meta=checkpoint_meta,
    plot_function=update_loss_plot,
    loss_plot_path=loss_plot_path
)
print("done:", model_path)