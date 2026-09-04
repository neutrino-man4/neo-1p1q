# qae_hep Codebase Audit: Control Flow and Cleanup Plan

Scope: core training pipeline only — `train.py`, `test.py` / `test_jetclass.py` / `test_copy.py`,
`utils.py` / `helpers/`, `quantum/`, `hydra_configs/`. Data/output/condor/docker/plotters/scratch
directories excluded per your instruction.

---

## 1. Control flow: `train.py` (the only entry point that matches the current architecture)

Invoked as `python train.py` with Hydra overrides, config resolved from
`hydra_configs/VQC/config.yaml` (composed on top of `hydra_configs/VQC/base.yaml`).

```
train.py: main(cfg)
│
├─ 1. Directory setup
│    save_dir = cfg.save_dir/cfg.seed ; plot_dir = save_dir/plots
│
├─ 2. wandb.init(project="1P1Q", config=cfg, name=f"{user}_{seed}")
│    writes save_dir/wandb_run_id.txt   [needed by test.py to resume the same run]
│
├─ 3. loguru logger.add(save_dir/logs.log)
│
├─ 4. ut.Pickle(cfg) + write save_dir/args.txt   (config snapshot, done TWICE — see 4.1)
│
├─ 5. Branch: cfg.resume?
│    ├─ True  → importlib-load saved_models/<seed>/FROZEN_ARCHITECTURE.py as `qc`,
│    │          reload last checkpoint under save_dir/checkpoints/ep*.pickle,
│    │          overwrite cfg with the pickled args from the original run
│    └─ False → import quantum.architectures as qc (live code)
│               freeze a copy of architectures.py, case_reader.py, losses.py into
│               save_dir/FROZEN_*.py (reproducibility snapshot)
│               delete save_dir/checkpoints/ if present
│
├─ 6. cost_fn = losses.probabilistic_loss  or  losses.VQC_cost   (cfg.loss)
│
├─ 7. VQC = quantum.architectures.QuantumClassifier(wires, shots, dev_name, layers, backend)
│    VQC.set_circuit(circuit_type=cfg.circuit_type)   # 'normal' or 'CNN'
│    NUM_WEIGHTS computed from wires/layers/extra_weights
│    init_weights = random uniform (unless resuming, then loaded checkpoint)
│
├─ 8. Data:
│    train/val filelists resolved via helpers.path_setter.PathSetter(cfg.data_dir)
│    train_loader, val_loader = case_reader.OneP1QDataLoader(..., dataset=cfg.dataset)
│      → dispatches to CASEDelphesJetDataset or CASEJetClassDataset
│
├─ 9. optimizer = qml.AdamOptimizer(cfg.lr)
│    trainer = quantum.architectures.QuantumTrainer(model=VQC, optimizer, loss_fn=cost_fn,
│                                                    wandb=wandb, logger=logger, ...)
│    trainer.set_directories(save_dir)   # creates save_dir/checkpoints
│
├─ 10. trainer.run_training_loop(train_loader, val_loader)
│      per epoch: train batches → optim.step_and_cost → wandb.log('train_loss')
│                 val batches   → loss_fn (eval mode)  → wandb.log('val_loss','val_auc')
│      early stopping on AUC plateau (with optional LR decay), checkpoint every epoch,
│      optimizer state (Adam moments) dumped to JSON alongside each checkpoint
│
└─ 11. finally: save history.pickle, plot loss curve to save_dir/history.png, wandb.finish()
```

Called modules and their role:
- `quantum/architectures.py` — `QuantumClassifier` (PennyLane QNode circuit + weight bookkeeping
  + Fisher-information utilities) and `QuantumTrainer` (the actual training loop, checkpointing,
  early stopping). This is where almost all real logic lives.
- `quantum/losses.py` — cost functions consumed by both `QuantumTrainer.iteration` and
  `QuantumClassifier.run_inference`.
- `quantum/math_functions.py` — small stateless helpers (MSE, BCE, two unused transforms).
- `case_reader.py` — `IterableDataset`s that stream `.h5` files into `(eta, phi, pt)` tensors.
- `helpers/utils.py` — feature-name/index lookups, pickling helpers, epoch-number parsing.
- `helpers/path_setter.py` — maps short dataset keys (`'VQC_train'`, `'grav_2p5_narrow'`, ...) to
  filesystem subpaths under `cfg.data_dir`.

## 2. Duplication and stale code found

| File | Status |
|---|---|
| `utils.py` (repo root) | **Fully superseded** by `helpers/utils.py`. Same `Pickle`/`Unpickle`/`folder_save`/`folder_load`/`print_events` functions, near-verbatim. `train.py` imports `helpers.utils`, never root `utils.py`. Nothing in the training path touches this file. |
| `test.py` | Calls `qc.QuantumAutoencoder(...)`, `qc.print_training_params()` (module-level call) and `loss.quantum_cost` / `loss.semi_classical_cost`. **None of these exist** in the current `quantum/architectures.py` (which defines `QuantumClassifier`, not `QuantumAutoencoder`) or `quantum/losses.py` (no `quantum_cost`). This script does not run against the current codebase — it's a leftover from an earlier autoencoder-based design. |
| `test_copy.py` | Same broken API surface as `test.py`, plus an `argparse`-based CLI predating the Hydra migration (no `cfg` object at all). Diff against `test.py` is almost entirely the arg-parsing front-end. |
| `test_jetclass.py` | 542 diff lines against `test.py` — effectively a third fork of the same script, specialized for the JetClass dataset instead of Delphes, with the same drift from the current `QuantumClassifier` API. |
| Config snapshot in `train.py` | Lines 45-48 and 132-135 write `args.pickle`/`args.txt` twice, identically, once before data loading and once after. Harmless but redundant. |
| `hydra_configs/` | 140+ leaf YAMLs (`VQC_JC_001_8Q7T.yaml` ... `VQC_JC_049_8Q.yaml`, similarly under `AOJ/`, `QAE/`, `QFI/`, `VPC/`) — one file per historical run, most differing from `base.yaml` by 2-3 values (seed, wires, trash_qubits). This is an experiment log encoded as config files, not a config system. |
| `quantum/math_functions.py` | `transform()` and `double_sided_leaky_relu()` are defined but not called anywhere in `losses.py` (both call sites are commented out). |
| `__pycache__/` dirs | Committed working-tree clutter (`quantum/__pycache__/*.pyc` for 3 Python versions) — already covered by `.gitignore`, just not cleaned locally. |

The three `test*.py` scripts appear to be inference/ROC-evaluation scripts that have not been
updated since `quantum/architectures.py` was refactored from an autoencoder (`QuantumAutoencoder`,
trash-qubit fidelity) to a classifier (`QuantumClassifier`, Hamiltonian expectation + BCE/MSE).
Right now none of them would run to completion.

## 3. Recommendations

### 3.1 Logging and experiment tracking → Weights & Biases

You already call `wandb.init`/`wandb.log`/`wandb.finish` in `train.py`, so the dependency is in
place; the gaps are in how it's used and in `loguru`/`print` sprawling everywhere alongside it.

- **Stop mixing `print()` and `logger.info()` for run narration.** `architectures.py` and
  `train.py` both use bare `print()` for things that belong in the log
  (`print_training_params`, `print_params`, checkpoint messages, the `time.sleep(3)` "Sleep on
  it... press CTRL-C" prompts). Route all of it through the `loguru` logger that's already passed
  in, and let wandb mirror the same log file via `wandb.init(..., sync_tensorboard=False)` +
  `wandb.save(save_dir/logs.log)` so a run's full log is attached to its wandb page.
- **Log config as wandb config, not as a redundant pickle+txt pair.** `wandb.init(config=...)`
  already gives you a versioned, filterable copy of every hyperparameter. Keep
  `ut.Pickle(cfg, 'args')` only if you need to reload `cfg` programmatically for `--resume`;
  drop the duplicate `args.txt` dump.
- **Use `wandb.log(..., step=epoch)` explicitly.** Right now `train_loss` is logged once per
  *batch* and `val_loss`/`val_auc` once per *epoch*, with no shared step counter — the two curves
  land on different x-axes in the dashboard. Pick one step semantic (e.g. global batch count) and
  pass it explicitly.
- **Log artifacts, not just scalars.** `FROZEN_ARCHITECTURE.py`, `FROZEN_LOSS.py`, the loss-curve
  PNG, and the final `trained_model.pickle` are already written to disk — register them with
  `wandb.log_artifact(...)` so a run's exact code and weights are one click away from its metrics,
  instead of only reachable via `save_dir` on the filesystem where the job ran.
- **Replace the `save_dir/wandb_run_id.txt` hand-rolled resume mechanism** with wandb's own run
  resume path (`wandb.init(id=..., resume="allow")`), which you're already halfway doing in the
  (currently broken) `test.py`. This removes one more bespoke text-file protocol.
- **`hydra_configs/` as your "experiment tracking"**: once wandb config-logging is relied on,
  most of the 140+ near-duplicate leaf YAMLs stop being load-bearing — they exist today because
  runs were tracked by which YAML you passed. Keep `hydra_configs/{VQC,QAE,VPC,QFI}/base.yaml` as
  the real templates, and either delete the run-specific leaves once their wandb runs exist, or
  move them under `hydra_configs/archive/` so `hydra_configs/` itself stays a small, readable set
  of templates plus CLI overrides (`python train.py wires=10 trash_qubits=8`) for one-off runs.

### 3.2 Modular, plug-and-play circuit design

Today `QuantumClassifier` hardcodes two circuit variants (`_vqc_circuit` / `_qcnn_implementation`)
as private methods selected by an `if circuit_type == 'CNN'` string switch inside `set_circuit`,
and each duplicates its own "normal" vs "state" (QFI) version of the same gate sequence by hand.
Adding a third circuit means editing this class in four places (both circuit + both state-circuit
variants) and extending the string switch.

Recommended shape:

```
quantum/circuits/
    base.py        # Circuit protocol: encode(inputs) -> None, ansatz(weights) -> None,
                    #                   n_weights(n_qubits, n_layers) -> int
    vqc.py          # today's _vqc_circuit, as a class implementing the protocol
    qcnn.py         # today's _qcnn_implementation, same protocol
    registry.py     # name -> class lookup, replaces the if/else in set_circuit
```

- Each circuit module owns **encoding** (data → rotations) and **ansatz** (trainable rotations +
  entangling layer) as separate methods. `QuantumClassifier` stops needing a bespoke
  `_vqc_state_circuit` twin for every circuit type — instead it wraps *any* registered circuit's
  `expval` version once, generically, to also produce a `qml.state()` version for QFI
  (`quantum_fisher`/`run_fisher_computation` don't care which circuit produced the state).
- `NUM_WEIGHTS` computation (currently duplicated between `train.py:115-117` and implicitly
  assumed inside each `_*_circuit`) becomes a `n_weights()` classmethod on each circuit, so adding
  a circuit can't silently desync the weight-count formula in `train.py` from the gate layout in
  `architectures.py`.
- New encodings (e.g. amplitude encoding instead of angle encoding) become new small classes
  rather than edits inside a 250-line method — this is the literal "operations as modules" ask.
- `set_circuit(circuit_type=...)` becomes `set_circuit(circuit_type=...)` → `registry.get(circuit_type)`,
  no code change needed to add an entry.

This is a moderate refactor (the two existing circuits move into the new files near verbatim); it
does not require touching `QuantumTrainer`, `losses.py`, or `case_reader.py`.

### 3.3 General cleanup / portability

1. **Delete `utils.py` (repo root)** — dead code, fully superseded by `helpers/utils.py`.
2. **Fix or retire `test.py`, `test_copy.py`, `test_jetclass.py`.** They reference a
   `QuantumAutoencoder` API that no longer exists. Either port one of them to
   `QuantumClassifier.run_inference` (which already exists and does the cost/score computation
   these scripts want) and delete the other two, or delete all three and write one fresh
   `evaluate.py` driven by the same Hydra config style as `train.py` (drop the argparse variant
   entirely — it predates the Hydra migration).
3. **Consolidate on Hydra config style across the whole pipeline** — `test_copy.py`'s argparse
   front-end is the only holdout; removing it (per #2) removes the last non-Hydra entry point.
4. **`quantum/math_functions.py`**: drop `transform()` and `double_sided_leaky_relu()` (unused,
   already commented out at both call sites) or wire them back in if they're mid-experiment.
5. **Prune `hydra_configs/`** per 3.1 — archive or delete superseded per-run leaf configs, keep
   `base.yaml` per project (`VQC/`, `QAE/`, `VPC/`, `QFI/`, `AOJ/`) as the maintained template.
6. **Remove committed `__pycache__/` directories** (`git rm -r --cached quantum/__pycache__` if
   they were ever committed, otherwise just `find . -name __pycache__ -exec rm -rf {} +` locally
   — they're already `.gitignore`d, just not cleaned).
7. **The `EOS`/`xrdcp` checkpoint-eviction path in `QuantumTrainer.run_training_loop`
   (`architectures.py:878-893`)** hardcodes CERN-specific env vars (`EOS_MGM_URL`,
   `CERN_USERNAME`, `BELLE2_EXEC`) inline in the generic training loop. Move this into an optional
   injected callback (e.g. `on_checkpoint: Callable | None` passed into `QuantumTrainer`) so the
   core trainer stays portable to a non-CERN environment and the site-specific eviction logic
   lives in one clearly-labeled place (or a `helpers/cern_eviction.py`).
8. **`helpers/utils.py` vs `case_reader.py` naming**: both define a local `getIndex`
   (`helpers/utils.py` and `helpers/path_setter.py` both define their own copy too — three copies
   of the same lookup function across the repo). Keep one in `helpers/utils.py`, import it
   everywhere else.
9. **Type hints are inconsistent** — `quantum/architectures.py` is fully typed (good, keep as the
   house style); `case_reader.py`, `helpers/*.py`, `quantum/losses.py` have none. Not urgent, but
   worth doing opportunistically when those files are next touched, per your existing CLAUDE.md
   Python formatting rule.

---

## Suggested order of operations

1. Delete root `utils.py` (zero risk, confirmed unused).
2. Decide the fate of `test.py` / `test_copy.py` / `test_jetclass.py` (fix-and-merge vs.
   rewrite-as-one) — this is the only genuinely broken part of the pipeline right now.
3. Switch `train.py`'s ad hoc logging/config-snapshotting over to wandb-native equivalents (3.1).
4. Do the circuit-module extraction (3.2) once a second circuit type is actually on the roadmap —
   no need to build the abstraction before there's a second concrete user of it.
5. Sweep the small items in 3.3 (cache dirs, unused functions, triplicated `getIndex`) as you go.
