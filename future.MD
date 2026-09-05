# Future Work

Things deliberately deferred rather than fixed now, with enough context to pick back up later.
Companion to `AUDIT.md`/`REFACTOR.md`/`refactored.MD`.

## Quantum Fisher Information (QFI) / metric-tensor computation

Removed entirely from the codebase on 2026-09-05 (`quantum_fisher()`, `run_fisher_computation()`,
the `_state_build`/`state_circuit_qnode`/`state_circuit()` machinery that served them,
`hydra_configs/QFI/`, `plotters/plot_utils.py`). Details and commit hashes: `refactored.MD`,
"D12 (attempted) — QFI removed entirely instead of fixed".

**Why it was removed instead of fixed:** it doesn't work under the currently pinned PennyLane
version (0.37.0), and the reason has nothing to do with this refactor's `CircuitWeights` change —
the pre-refactor flat-weight-vector code fails identically. Root cause, as far as it was traced:
`VQCCircuit.encode()`'s `if inputs.shape[0] == 1: ... .item()` branch (needed for the normal
training/inference path, where a single sample is fed one at a time) breaks under
`qml.metric_tensor`/`qml.adjoint_metric_tensor`'s internal tracing, which wraps classical circuit
construction in autograd `ArrayBox`es even for arguments not being differentiated in that pass.
Removing the `.item()` calls gets past that specific crash but immediately surfaces at least two
more, distinct, un-root-caused errors:
- `qml.metric_tensor(..., approx=None)` needs the device to carry an extra auxiliary wire that
  `quantum_fisher()`'s own construction path didn't visibly set up for every case.
- `qml.metric_tensor(..., approx='block-diag')` and `qml.adjoint_metric_tensor` both then fail with
  `AttributeError: 'tuple' object has no attribute 'shape'`, cause unknown.

**What re-implementing it later will need:**
1. Root-cause and fix the `.item()`/tracing incompatibility in `encode()` — likely means not
   converting single-sample batch dimensions to Python scalars via `.item()`, and instead letting
   gate parameters stay as 0-d arrays (the plain-call output was confirmed identical either way,
   `onp.allclose` on the state vector, so this is very likely fine, but wasn't tested through a
   full forward+backward training run).
2. Root-cause the remaining "no free auxiliary wire" and "'tuple' object has no attribute 'shape'"
   errors — these appeared only after fixing (1), so they're two separate problems layered under
   the first.
3. Once QFI is differentiable again, use `Circuit.build()`'s `measure_override` parameter (kept in
   `quantum/circuits/base.py` specifically for this — it never mentioned QFI and cost nothing to
   keep) to rebuild a state-circuit the same way D6 originally did:
   ```python
   def _state_build(self, weights, inputs=None, wires=None):
       wires = wires if wires is not None else self.auto_wires
       return self._impl.build(weights, inputs, wires, measure_override=lambda *_: qml.state())
   ```
4. Differentiate `metric_tensor`/`adjoint_metric_tensor` w.r.t. `rot` only, with `aux` baked in as
   constants (not part of the trainable argument) — a quick spike during D12's verification showed
   this reproduces the pre-refactor rotation-only QFI submatrix exactly, once the tracing crash
   above is actually fixed, so it doesn't need re-deriving from scratch.
5. Decide what to do with the currently-dead-but-commented-out qubit-reindexing code inside the old
   `quantum_fisher()` (never actually applied — `return rotation_qfi#reordered_qfi`) if the
   original ordering convention still matters.
