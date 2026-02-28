#!/usr/bin/env bash
set -euo pipefail

python src/sim/convert.py \
  --cosmo-txt data/latin_hypercube_params.txt \
  --in-root   /path/to/quijote/latin_hypercube_HR \
  --out-root  outputs/latin_hypercube_hr \
  --sim-min 0 --sim-max 2000

python src/sim/brightness_temperature.py \
  --base-input outputs/latin_hypercube_hr \
  --latin-file data/latin_hypercube_params_full.txt \
  --output-base outputs \
  --sim-min 0 --sim-max 2000

python src/sim/power_spectrum.py \
  --input-root outputs \
  --output-root outputs/power_spectra \
  --sim-min 0 --sim-max 2000

python src/ml/data_preparation_2c.py \
  --power-root outputs/power_spectra \
  --fits-root outputs/fits \
  --latin-file data/latin_hypercube_params.txt \
  --outdir datasets/825_6145 \
  --stage all

python src/ml/train_2c_snpe.py \
  --dataset-dir datasets/825_6145 \
  --out-dir snpe_outputs/825_6145 \
  --seeds 2 \
  --epochs 200 \
  --batch-size 50
