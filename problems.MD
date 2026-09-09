# Known problems

This is the active development backlog for the supported JetClass VQC pipeline. Completed work
belongs in Git history rather than this file. Status checked against the code on 2026-09-09.

## P4: resume is incomplete

Resume now reads the saved YAML before writing, extracts structured weights, and uses the shared
circuit factory. It still starts a fresh optimizer, resets the training loop to epoch zero, and
selects checkpoints lexicographically. Epoch checkpoints lack the final model's provenance.

Select checkpoints by numeric epoch, verify their configuration/source provenance, and restore
optimizer state, history, and the next epoch. Resume currently warns about the fresh optimizer.

Done when a saved run can resume for another training step with its original configuration,
weights, optimizer state, and circuit implementation.

## P5: `probabilistic_loss` does not match the VQC output

The registered VQC circuit returns one Hamiltonian expectation value per sample.
`probabilistic_loss` indexes that result as a two-dimensional class-probability array, so
`loss=prob` raises `IndexError`.

Choose one supported meaning for this mode:

- Remove `probabilistic_loss` and reject `loss=prob` during configuration validation.
- Convert the scalar expectation value into a documented binary probability.
- Add a circuit whose measurement returns class probabilities.

Done when the selected behavior has an end-to-end training and validation test.

## P6: optimizer checkpoints are unusable

`QuantumTrainer.save()` passes Adam's NumPy moment arrays directly to `json.dump`, which cannot
serialize them. The initial checkpoint also tries to join a path with `opt_name=None`. Both errors
are caught and printed, so weight checkpoints are written without usable optimizer state.

Use a serialization format that preserves all optimizer fields, assign a valid filename for every
checkpoint, and load the state in the resume path described in P4. Do not swallow serialization
errors.

Done when each weight checkpoint has matching optimizer state and a resumed step matches an
uninterrupted run from the same state.

## P10: the loader reads and converts more data than requested

`CASEJetClassDataset._load_file()` reads all rows from each HDF5 dataset before `_read_class()`
selects the requested count. A partial run can therefore read a full 100,000-jet file for a much
smaller sample.

Pass the remaining event count into `_load_file()` and slice the HDF5 datasets before conversion.
Then store inputs as plain NumPy `float32`, remove the Torch round trip in `__iter__`, and remove or
bypass the redundant `DataLoader(batch_size=None)` wrapper.

Done when a partial read touches only the requested HDF5 rows and iteration returns the same
values and labels without the Torch conversion chain.

## P13: installation metadata is incomplete

`train.py` imports OmegaConf and WandB, but neither package is listed in `requirements.txt`.
`setup.sh` therefore does not create a complete runtime environment from the repository alone.

Add every direct runtime dependency to the environment definition and verify a clean installation
can import both entry points.

## P14: unused configurations remain

The AOJ, QAE, VPC, top-level, and JetClass base configurations lack fields required by the current
entry points. Delete them if those workflows are retired. Otherwise, give each configuration a
supported entry point and a smoke test.

## P15: pickle writes leak file handles

`helpers.utils.Pickle()` opens pickle files without closing them. The test suite reports
`ResourceWarning` for checkpoint and history writes.

Use context managers for all file operations and avoid changing the process working directory
during serialization.

## P7: Quantum Fisher Information is deferred

QFI and metric-tensor computation are not part of the current classifier API. The remaining
technical work is recorded in `future.MD`.
