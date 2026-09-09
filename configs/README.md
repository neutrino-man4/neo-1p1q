# Configuration

`base.yaml` is the default configuration for the JetClass variational quantum classifier. Both `train.py` and `evaluate.py` use the same schema. Training saves the resolved configuration with the model so that evaluation can rebuild the same circuit and data selection.

## Loading and overriding values

The project uses OmegaConf to read YAML and apply command line overrides. YAML provides the defaults, while arguments in `key=value` form replace individual entries:

```bash
python train.py \
  --config configs/base.yaml \
  seed=run_001 \
  data_dir=/path/to/JetClass \
  save_dir=/path/to/saved_models \
  epochs=40 \
  shots=-1
```

OmegaConf parses the value type from the command line. For example, `epochs=40` is an integer, `flat=true` is a Boolean, and `lr=0.01` is a float. Nested entries use dot notation:

```bash
python train.py aux_weights.bias=0.0 aux_weights.scale_factor=1.5
```

An entry that is not already in the YAML can also be supplied. This is useful for supported optional settings such as an explicit particle count:

```bash
python train.py seed=run_002 num_particles=12
```

Use `--print-config` to inspect the merged configuration without starting training:

```bash
python train.py --config configs/base.yaml epochs=5 --print-config
```

Avoid spaces around `=`. Quote an entire override when its value contains spaces or shell characters.

## Run and storage entries

| Entry | Purpose |
| --- | --- |
| `seed` | Names the run directory. Training converts it to a string and saves the run under `<save_dir>/<seed>/`. Use a new value for each run. |
| `data_dir` | Root of the JetClass dataset. Relative paths are resolved when training starts. |
| `save_dir` | Parent directory for saved runs. Each seed gets its own subdirectory. |
| `dump` | Parent directory for evaluation results. Evaluation writes to `<dump>/<seed>/`. |
| `desc` | Short run description passed to Weights & Biases. |
| `save` | Enables checkpoints, history, and final model output. Keep this `true` for resumption or evaluation. |
| `resume` | Requests resumption from the latest checkpoint. Prefer the `--resume` flag described below. |

The default paths are relative to the directory where the command is run. Override them when the data or model storage is elsewhere.

## Data entries

| Entry | Purpose |
| --- | --- |
| `signal` | JetClass sample directory assigned label 1. |
| `background` | JetClass sample directory assigned label 0. |
| `n_signal`, `n_background` | Number of signal and background jets requested for training. |
| `n_signal_val`, `n_background_val` | Number of jets requested for validation. |
| `n_signal_test`, `n_background_test` | Number of jets requested for evaluation. |
| `batch_size` | Number of jets in each training and validation batch. The last smaller batch is retained. |
| `flat` | Uses `flat_train`, `flat_val`, and `flat_test` when `true`; otherwise uses `train`, `val`, and `test`. |
| `norm_pt` | Divides constituent transverse momentum by the jet transverse momentum when `true`. When `false`, the loader applies its fixed scaling limits. |

The loader expects this directory structure:

```text
<data_dir>/<split>/<signal>/*.h5
<data_dir>/<split>/<background>/*.h5
```

Each HDF5 file must contain `jetConstituentsList` and `jetFeatures`. Constituents must be ordered by decreasing transverse momentum.

The circuit consumes `wires * num_layers` particles per jet by default. Set `num_particles` to load more, but it cannot be smaller than that product.

## Circuit entries

| Entry | Purpose |
| --- | --- |
| `wires` | Number of data qubits. |
| `num_layers` | Number of encoding, entangling, and trainable rotation layers. Each layer consumes another group of `wires` particles. |
| `shots` | Number of measurement shots. A nonpositive value selects analytic expectation values. |
| `device_name` | PennyLane device name. The default is the fast CPU simulator `lightning.qubit`; `default.qubit` is useful for debugging small runs. |
| `backend` | PennyLane QNode interface. The maintained training path uses `autograd`. |
| `circuit_type` | Circuit name from `quantum/circuits/registry.py`. The supported value is `normal`. |
| `operations_per_qubit` | Trainable rotation parameters per qubit and layer. The current VQC requires `3` for its RZ, RY, and RX rotations. |
| `aux_weights.scale_factor` | Initial trainable scale applied during angle encoding. |
| `aux_weights.bias` | Initial trainable bias added to the circuit output before the loss is calculated. |

These entries define the circuit and weight shapes. Changing them when evaluating an existing run would produce a different model, so evaluation does not accept overrides.

## Optimization entries

| Entry | Purpose |
| --- | --- |
| `epochs` | Maximum number of training epochs. Validation also runs once before the first update. |
| `lr` | Initial Adam learning rate. |
| `loss` | Classifier loss. Supported values are `MSE` and `BCE`. |
| `improv` | Minimum validation AUC improvement required by the stopping logic after its initial warmup. |
| `lr_decay` | Halves the learning rate after insufficient improvement when `true`. When `false`, insufficient improvement stops training. |
| `patience` | Maximum number of learning rate reductions before early stopping when `lr_decay=true`. |

## Saved configurations

For a new run, training merges the selected YAML with every command line override, resolves the paths, and writes the result to:

```text
<save_dir>/<seed>/config.yaml
```

The run directory also contains the circuit source used for that run, checkpoints, and `trained_model.pickle`. The saved YAML is part of the model record. Do not replace it with the current `configs/base.yaml` after the run has started.

Resume from the saved settings and latest checkpoint with:

```bash
python train.py --config /path/to/saved_models/run_001/config.yaml --resume
```

Resumption reloads the original saved configuration. Training option overrides supplied with `--resume` do not replace the recorded settings.

Evaluation accepts a saved run configuration directly:

```bash
python evaluate.py --config /path/to/saved_models/run_001/config.yaml
```

It can also reconstruct that path from the run seed and model directory:

```bash
python evaluate.py --seed run_001 --model-dir /path/to/saved_models
```

Evaluation verifies the saved configuration, circuit files, and final weights before running inference. It does not use an epoch checkpoint as a substitute for `trained_model.pickle`.
