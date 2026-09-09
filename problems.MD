# Known problems

This is the active development backlog for the supported JetClass VQC pipeline. Completed work
belongs in Git history rather than this file. Status checked against the code on 2026-09-09.

## P15: pickle writes leak file handles

`helpers.utils.Pickle()` opens pickle files without closing them. The test suite reports
`ResourceWarning` for checkpoint and history writes.

Use context managers for all file operations and avoid changing the process working directory
during serialization.

## P7: Quantum Fisher Information is deferred

QFI and metric-tensor computation are not part of the current classifier API. The remaining
technical work is recorded in `future.MD`.
