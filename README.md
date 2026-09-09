# 1P1Q

1P1Q trains and evaluates a variational quantum classifier on JetClass HDF5 data.

## Installation

Create the project environment and install its dependencies:

```bash
conda create -n pennylane-gpu-sep2026 python=3.10 -y
conda activate pennylane-gpu-sep2026
python -m pip install -r requirements.txt
```

Training uses Weights & Biases. Set `WANDB_API_KEY` for online logging or run locally with:

```bash
export WANDB_MODE=offline
```

## Data

Arrange the JetClass files by split and class:

```text
<data_dir>/
├── train/
│   ├── TTBar_/*.h5
│   └── ZJetsToNuNu/*.h5
├── val/
│   ├── TTBar_/*.h5
│   └── ZJetsToNuNu/*.h5
└── test/
    ├── TTBar_/*.h5
    └── ZJetsToNuNu/*.h5
```

Each file must contain `jetConstituentsList` and `jetFeatures`. Constituents must be ordered by decreasing transverse momentum. For flattened samples, use `flat_train`, `flat_val`, and `flat_test` and set `flat=true`.

## Training

Pass configuration changes as `key=value` arguments:

```bash
python train.py \
  --config configs/VQC/base.yaml \
  seed=run_001 \
  data_dir=/path/to/JetClass \
  save_dir=/path/to/saved_models
```

The run is saved under `<save_dir>/<seed>/`. It contains the resolved `config.yaml`, a copy of the circuit source, epoch checkpoints, final weights, logs, history, and plots.

Resume the latest checkpoint for a run with:

```bash
python train.py --config /path/to/saved_models/run_001/config.yaml --resume
```

## Evaluation

Evaluate the final saved model with its run configuration:

```bash
python evaluate.py --config /path/to/saved_models/run_001/config.yaml
```

The same run can be selected by seed:

```bash
python evaluate.py --seed run_001 --model-dir /path/to/saved_models
```

Evaluation loads the saved circuit and `trained_model.pickle`. It writes `test_results.pickle` to `<dump>/<seed>/` and `plots/roc_curve.png` to the run directory. The `dump` path comes from the saved configuration.
