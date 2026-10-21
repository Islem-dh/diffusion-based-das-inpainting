"""RAD v3: trained from scratch for 40 epochs with EMA (0.999), masked MSE, phase-1 probability 0.75
and input scaling (normalized data * data_scale, global train std -> 0.5; see notebooks/scaling_factor.ipynb).
Needs checkpoints/data_scaling.json.
Usage:  python scripts/train_rad_v3.py   (resumes from the v3 checkpoint if it exists)"""

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
from ddpm.training.rad_training3 import train_rad

device="cuda" if torch.cuda.is_available() else "cpu"

das_data=DAS_DATA
train_loader_earthquake=DataLoader(DasDataset([das_data/"train_1001.h5"]),batch_size=16,shuffle=True)
test_loader_earthquake=DataLoader(DasDataset([das_data/"test_1001.h5"]),batch_size=16,shuffle=False)

scheduler_rad=RADNoiseScheduler(ph1_steps=1000,ph2_steps=1000,beta_start=1e-4,beta_end=0.02,device=device)

checkpoints=CHECKPOINTS
with open(checkpoints/"data_scaling.json") as f: scaling=json.load(f)
data_scale=scaling["data_scale"]      # 2.023
clip_value=scaling["clip_value"]      # 2.397 
print("data_scale:",data_scale,"| clip_value:",clip_value)

checkpoint_meta={
    "base_channels":64,"channels_multiplier":(1,1,2,2,4),
    "normalization":{"function":"arcsinh","s":50.0,"A_max":12350.0},
    "data_scaling":scaling,
    "model_key":"model_rad","optimizer_key":"optimizer_rad","train_loss_key":"train_losses","val_loss_key":"val_losses"
}

model_rad=RAD(in_channels=1,out_channels=1,base_channels=64,channels_multiplier=(1,1,2,2,4),nbr_rb=2,dropout=0.1).to(device)
optimizer_rad=torch.optim.Adam(model_rad.parameters(),lr=1e-4)

model_path_v3=checkpoints/"das_rad_earthquake_model_v3.pth"
loss_plot_path_v3=checkpoints/"das_rad_earthquake_loss_v3.png"

if model_path_v3.exists():
    ckpt=torch.load(model_path_v3,map_location=device)
    model_rad.load_state_dict(ckpt["model_rad"])
    optimizer_rad.load_state_dict(ckpt["optimizer_rad"])
    ema_rad=copy.deepcopy(model_rad); ema_rad.load_state_dict(ckpt["ema_model_rad"])
    train_losses=ckpt.get("train_losses",[]); val_losses=ckpt.get("val_losses",[])
    start_epoch=ckpt["completed_epochs"]
    print("resuming v3 from epoch",start_epoch)
else:
    ema_rad=copy.deepcopy(model_rad)     
    train_losses=[]; val_losses=[]
    start_epoch=0
    print("training v3 from scratch")

epochs=40   
train_losses,val_losses,ema_rad=train_rad(
    model=model_rad,
    train_loader=train_loader_earthquake,
    val_loader=test_loader_earthquake,
    scheduler=scheduler_rad,
    optimizer=optimizer_rad,
    device=device,
    epochs=epochs,
    checkpoint_path=model_path_v3,
    model_key="model_rad",
    optimizer_key="optimizer_rad",
    train_loss_key="train_losses",
    val_loss_key="val_losses",
    start_epoch=start_epoch,
    train_losses=train_losses,
    val_losses=val_losses,
    checkpoint_meta=checkpoint_meta,
    plot_function=update_loss_plot,
    loss_plot_path=loss_plot_path_v3,
    ema_model=ema_rad,
    ema_decay=0.999,
    phase1_prob=0.75,
    masked_loss=True,
    data_scale=data_scale
)
print("done:", model_path_v3)