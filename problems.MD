# Known problems

This is the active development backlog for the supported JetClass VQC pipeline. Completed work
belongs in Git history rather than this file. Status checked against the code on 2026-09-09.

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
