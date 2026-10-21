"""RAD v2: fine-tuning of RAD v1 for 20 more epochs (60 in total) with EMA (0.999),
masked MSE (only noised pixels) and phase-1 sampling probability 0.75.
Needs the v1 checkpoint (scripts/train_rad_v1.py).
Usage:  python scripts/train_rad_v2.py   (resumes from the v2 checkpoint if it exists)"""

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
from ddpm.training.rad_training2 import train_rad

device="cuda" if torch.cuda.is_available() else "cpu"

das_data=DAS_DATA
train_loader_earthquake=DataLoader(DasDataset([das_data/"train_1001.h5"]),batch_size=16,shuffle=True)
test_loader_earthquake=DataLoader(DasDataset([das_data/"test_1001.h5"]),batch_size=16,shuffle=False)

scheduler_rad=RADNoiseScheduler(ph1_steps=1000,ph2_steps=1000,beta_start=1e-4,beta_end=0.02,device=device)

checkpoints=CHECKPOINTS
checkpoint_meta={
    "base_channels":64,"channels_multiplier":(1,1,2,2,4),
    "normalization":{"function":"arcsinh","s":50.0,"A_max":12350.0},
    "model_key":"model_rad","optimizer_key":"optimizer_rad","train_loss_key":"train_losses","val_loss_key":"val_losses"
}

model_rad=RAD(in_channels=1,out_channels=1,base_channels=64,channels_multiplier=(1,1,2,2,4),nbr_rb=2,dropout=0.1).to(device)
optimizer_rad=torch.optim.Adam(model_rad.parameters(),lr=1e-4)

model_path_old=checkpoints/"das_rad_earthquake_model.pth"
model_path_v2=checkpoints/"das_rad_earthquake_model_v2.pth"
loss_plot_path_v2=checkpoints/"das_rad_earthquake_loss_v2.png"

if model_path_v2.exists():
    ckpt=torch.load(model_path_v2,map_location=device)
    model_rad.load_state_dict(ckpt["model_rad"])
    optimizer_rad.load_state_dict(ckpt["optimizer_rad"])
    ema_rad=copy.deepcopy(model_rad); ema_rad.load_state_dict(ckpt["ema_model_rad"])
    train_losses=ckpt.get("train_losses",[]); val_losses=ckpt.get("val_losses",[])
    start_epoch=ckpt["completed_epochs"]
    print("resuming v2 from epoch",start_epoch)
else:
    ckpt=torch.load(model_path_old,map_location=device)
    model_rad.load_state_dict(ckpt["model_rad"])
    optimizer_rad.load_state_dict(ckpt["optimizer_rad"])
    ema_rad=copy.deepcopy(model_rad)
    train_losses=[]; val_losses=[]
    start_epoch=ckpt.get("completed_epochs",40)
    print("starting v2 from old checkpoint, epoch",start_epoch)

epochs=60   # total; 20 more epochs after 40
train_losses,val_losses,ema_rad=train_rad(
    model=model_rad,
    train_loader=train_loader_earthquake,
    val_loader=test_loader_earthquake,
    scheduler=scheduler_rad,
    optimizer=optimizer_rad,
    device=device,
    epochs=epochs,
    checkpoint_path=model_path_v2,
    model_key="model_rad",
    optimizer_key="optimizer_rad",
    train_loss_key="train_losses",
    val_loss_key="val_losses",
    start_epoch=start_epoch,
    train_losses=train_losses,
    val_losses=val_losses,
    checkpoint_meta=checkpoint_meta,
    plot_function=update_loss_plot,
    loss_plot_path=loss_plot_path_v2,
    ema_model=ema_rad,
    ema_decay=0.999,
    phase1_prob=0.75,
    masked_loss=True
)
print("done:", model_path_v2)