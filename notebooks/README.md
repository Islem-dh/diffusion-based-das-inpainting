# Notebooks

Run from this folder with the package installed (`pip install -e .`). Model weights are expected in `checkpoints/`.

| Notebook | Content |
|---|---|
| **Data** | |
| `DAS_Viewer` | DAS patches of the NPME classes |
| `DAS_Normalization` | Amplitude statistics, arcsinh normalization |
| `DAS_ddpm_shifting_norm` | Normalization experiments for DDPM training |
| **Priors (DDPM)** | |
| `DAS_ddpm_unet_model` | Baseline U-Net |
| `DAS_ddpm_unet_model_ssa` / `_spcii` / `_mamba` | Priors on 1001, generation quality |
| `Prior_50_50_normal_mse` | Mamba prior on 50 % noise + 50 % earthquake |
| `Prior_10_90_*`, `Prior_30_70_*`, `Prior_50_50_weighted_mse` | Other data mixes and weighted MSE |
| `DAS_mamba_weighted`, `DAS_mamba_weighed_mse`, `DAS_mamba_region_weighed_mse*` | Loss weighting experiments |
| **RAD** | |
| `train_rad` | RAD v1, RDDIM sampling, x0 clipping |
| `test_rad_masks` | RAD training masks |
| `ema_rad` | RAD v2 (EMA, masked MSE, phase-1 focus) |
| `scaling_factor` | Input scaling factor (`data_scaling.json`) |
| `ema_rad_scaled` | RAD v3 (input scaling), comparison with v1/v2 |
| `rad_traffic_removal` | RAD on real traffic patches |
| **RePaint** | |
| `repaint_results` | RePaint with SSA, SPCII and Mamba priors |
| **Evaluation** | |
| `inpainting_eval` | RAD vs. RePaint: ground-truth test (1001) and real traffic removal (1111) |
| `geo_rad` | RAD on a 30 s segment over the whole cable |

`archive/`: early MNIST experiments, not maintained.

Training of the final models: see `scripts/` and the main README.
