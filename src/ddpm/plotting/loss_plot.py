"""Utilities for plotting DDPM  and RAD loss plots."""

import matplotlib.pyplot as plt
from IPython.display import clear_output

def update_loss_plot(train_losses,val_losses, loss_plot_path):
    """Update and save the training and validation loss curves."""

    clear_output(wait = True)
    plt.figure(figsize = (6,4))
    plt.plot(range(1,len(train_losses)+1), train_losses, marker="o", label="Training Loss")
    plt.plot(range(1,len(val_losses)+1), val_losses, marker="o", label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Noise Prediction Loss")
    plt.title("RAD DAS Loss")
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.show() 
