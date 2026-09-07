# 1P1Q: particle physics data encoding for jet physics

<img src="https://etpwww.etp.kit.edu/~abal/icons/1p1q_logo.png" alt="1P1Q logo" width="15%">

**Authors:** Aritra Bal¹, Benedikt Maier², Melik Oughton², Eric Pezone²  
¹ Karlsruhe Institute of Technology (KIT), Germany  
² Imperial College London, UK

[![arXiv](https://img.shields.io/badge/arXiv-2502.17301-b31b1b.svg)](https://arxiv.org/abs/2502.17301)

This repository trains and evaluates a variational quantum classifier on JetClass data. The
supported pipeline is binary classification with the `normal` VQC circuit. It uses OmegaConf for
configuration and Weights & Biases for run tracking.

## Environment

The reference environment uses Python 3.10 and PennyLane 0.37.0. The test suite also runs with
Python 3.14, PennyLane 0.45.1, and NumPy 2.

Create the reference Conda environment:

```bash
bash setup.sh
conda activate quantum-cpu
pip install omegaconf wandb
export PYTHONPATH="$PYTHONPATH:$PWD"
```

`omegaconf` and `wandb` are direct runtime dependencies but are not yet listed in
`requirements.txt`.

Set `WANDB_API_KEY` before training, or use `WANDB_MODE=offline` for a local run.

## Data layout

The loader expects separate signal and background directories for each split:

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

With `flat=true`, the split names are `flat_train`, `flat_val`, and `flat_test`. Each HDF5 file
must contain `jetConstituentsList` and `jetFeatures`. Constituents must be sorted by decreasing
particle transverse momentum.

## Training

`configs/VQC/base.yaml` is the maintained configuration. Override values with OmegaConf dotlist
arguments:

```bash
python train.py \
  --config configs/VQC/base.yaml \
  seed=run_001 \
  base_dir="$PWD" \
  data_dir=/path/to/JetClass \
  save_dir=/path/to/saved_models \
  batch_size=100
```

Training and validation broadcast each minibatch through the circuit and reduce its loss to the
mean over that minibatch. A smaller final batch is included in the epoch metrics with its actual
sample count.

Useful overrides include:

- `signal` and `background`: JetClass sample directory names.
- `n_signal`, `n_background`: training events per class.
- `n_signal_val`, `n_background_val`: validation events per class.
- `wires`: number of qubits. By default, the loader keeps `wires * num_layers` particles per jet.
- `num_particles`: optional larger particle count; it must be at least `wires * num_layers`.
- `shots=-1`: analytic expectation values instead of finite-shot evaluation.
- `device_name`: PennyLane device, such as `default.qubit`, `lightning.kokkos`, or
  `lightning.gpu` when the corresponding plugin is installed.

Every run writes its merged configuration to `<save_dir>/<seed>/config.yaml`, along with logs,
checkpoints, and frozen source files.

## Evaluation

Evaluate a saved run with its recorded configuration:

```bash
OMP_NUM_THREADS=1 OMP_PROC_BIND=spread \
python evaluate.py --config /path/to/saved_models/run_001/config.yaml
```

Evaluation loads `trained_model.pickle`, or the latest epoch checkpoint if the final model is
absent. It writes scores and labels to `<dump>/<seed>/test_results.pickle` and the ROC curve to
`<save_dir>/<seed>/plots/roc_curve.png`.

## Tests

Run the suite with either supported environment:

```bash
conda run -n quantum-cpu python -m unittest discover -s tests -q
conda run -n pennylane-gpu-sep2026 python -m unittest discover -s tests -q
```

Some tests currently reproduce known failures and pass when those failures occur. The active
backlog identifies those cases.

## Main files

- `train.py`: training entry point.
- `evaluate.py`: inference and ROC evaluation.
- `case_reader.py`: balanced JetClass loader.
- `helpers/config.py`: OmegaConf loading, overrides, and config snapshots.
- `quantum/architectures.py`: classifier and training loop.
- `quantum/circuits/`: circuit protocol, registry, and VQC implementation.
- `quantum/losses.py`: classifier loss functions.

## Citation

```bibtex
@article{bal2025anomaly,
  title={Anomaly detection in high-energy physics using a quantum autoencoder},
  author={Bal, Aritra and Maier, Benedikt and Oughton, Melik and Pezone, Eric},
  journal={arXiv preprint arXiv:2502.17301},
  year={2025}
}
```
