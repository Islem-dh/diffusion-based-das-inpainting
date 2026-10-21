"""Central project paths. Every notebook and script imports its paths from here,
so they keep working no matter which folder they are run from."""

import os
from pathlib import Path

# project root = folder that contains src/ (src/ddpm/paths.py -> parents[2])
PROJECT_ROOT = Path(os.environ.get("DDPM_ROOT", Path(__file__).resolve().parents[2]))

CHECKPOINTS = PROJECT_ROOT/"checkpoints"            # model weights (.pth), data_scaling.json
RESULTS = PROJECT_ROOT/"results"                    # saved inpainting results, tables (.pt, .json)
OUTPUTS = PROJECT_ROOT/"outputs"                    # figures, data analysis
CONFIGS = PROJECT_ROOT/"configs"                    # training configs (.yaml)

# datasets on the server
DAS_DATA = Path(os.environ.get("DAS_DATA", "/datasets/training/earthquake/share/SISSLE_south30"))    # train/test_NPME.h5 patches
DAS_RAW_100HZ = Path(os.environ.get("DAS_RAW", "/datasets/training/earthquake/SISSLE_100Hz/south30"))  # raw 1-min recordings

for p in (CHECKPOINTS, RESULTS, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)