# 21cm Signals Simulation Based Inference (SBI) Pipeline 

Reliable probabilistic inference framework for estimating cosmological parameters at z = 1 using high-dimensional 21cm brightness-temperature maps derived from Quijote halo catalogs.

This repository implements a reproducible pipeline: from forward modeling physical simulations to training Deep Learning models for sequential neural posterior estimation (SNPE).

**Forward Modeling**: Converts Quijote Friends-of-Friends (FoF) catalogs into high-resolution HEALPix brightness-temperature maps and extracts angular power spectra using NaMaster.

**Deep Learning & SBI**: Utilizes PyTorch and the sbi library to perform robust uncertainty quantification and parameter estimation.

**Multiple Data Representations**: Supports both 2-channel summary statistics (CNN-based embeddings) and full field-level HEALPix maps (DeepSphere spherical CNN embeddings).

Reproducibility: Environment configurations and strict separation of data/code to run efficiently on HPC clusters.

**Note:** Simulation inputs and generated outputs (**FITS/NPZ/checkpoints**) are not included in this repository.

## Repository Structure

```text
src/
├── sim/                      # forward model / data creation
│   ├── convert.py            # Quijote FoF → halos.txt
│   ├── brightness_temperature.py
│   │   # halos.txt → HEALPix maps (Tb, dTb) + metadata
│   └── power_spectrum.py     # maps → binned angular power spectra (NaMaster)
│
└── ml/                       # ML / SBI
    ├── data_preparation_2c.py
    │   # 2-channel dataset from power spectra (+ mean Tb)
    ├── train_2c_snpe.py      # SNPE training with CNN embedding (2-channel)
    ├── data_fieldlevel_healpix.py
    │   # field-level dataset (HEALPix dTb maps)
    └── train_field_snpe.py   # SNPE training with DeepSphere embedding
```
## Installation

### Option A (conda, recommended)

```bash
conda env create -f environment.yml
conda activate 21cm-sbi
```

### Option B (pip)
```bash
pip install -r requirements.txt
```

### Environment
```bash
Python 3.12.7 (conda-forge)
numpy 1.26.4
scipy 1.16.2
torch 2.5.1+cu124
sbi 0.25.0
healpy 1.18.1
pymaster (NaMaster) 2.3.2

#Install the CUDA-enabled PyTorch build recommended for your system if needed.

License
This project is licensed under the MIT License - see the  file for details.
