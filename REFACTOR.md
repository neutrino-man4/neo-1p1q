# Circuit Module Refactor

Companion to `AUDIT.md`, section 3.2 ("modular, plug-and-play circuit design"). Part A is
background for the design. Part B covers the remaining deliverables, ordered by dependency.

**Status:** D1, D2, D4, D5, D6, D7, D8, D9, D11 are implemented — see `refactored.MD`. This file
covers what's left: D10 and D12 (proposed, not yet decided).

**Scope of this pass: VQC only.** QCNN (`_qcnn_implementation`/`_qcnn_state_circuit`,
`circuit_type='CNN'`) is removed from the live code as part of D9, not merely left unmigrated.
Reintroducing it later is a separate deliverable, following the same pattern as D2 once there's a
concrete second circuit again (per `AUDIT.md`'s own advice: don't build for a second circuit type
before one actually exists).

---

## Part A — Background

### A1. Goal

`quantum/architectures.py` hardcodes `_vqc_circuit` as a private method of `QuantumClassifier`,
with a hand-duplicated `expval` version and `state()` version (for QFI). Adding a circuit today
means editing `QuantumClassifier` directly and extending a string switch in `set_circuit()`.

Target: a circuit is a class implementing a small `Circuit` protocol, registered by name.
`QuantumClassifier` and `QuantumTrainer` do not change when a circuit is added later.

### A2. The four operations

VQC's circuit body decomposes into four steps, each a natural unit of reuse for whatever circuit
comes next:

- **encode** — state preparation from data; re-run every layer, full wire set, offset into
  `inputs` by `w + L*N`.
- **entangle** — fixed `CNOT` ring over all N wires, identical every layer.
- **rotate** — one trainable weight triplet (RZ/RY/RX) per wire, applied after entangle.
- **measure** — scalar `expval` of a weighted `Hamiltonian` over all wires.

Their **composition** — the per-layer loop calling encode → entangle → rotate, then measuring once
at the end — is `build()`, with a default implementation covering exactly this shape. `build()`
stays overridable on the protocol (zero cost today) so a future circuit whose loop shape genuinely
differs isn't blocked by this one — but no such circuit is in scope here.

### A3. Weight representation

Today's flat weight vector mixes two unrelated things via magic negative indices: per-qubit
rotation angles (`weights[start:start+3*N:3]`-style slicing) and auxiliary scalars — scale factor,
loss bias, Hamiltonian coefficients — stuffed into the tail (`weights[-1]`, `weights[-6:-1]`,
`weights[-11:-1]`, one of them hardcoded a second time in `losses.py`). Replaced (D1/D2, done) with
two explicit pieces:

- **Rotation tensor**, shape `(L, N, R)` — `L = cfg.num_layers`, `N` = qubit count, both already in
  config. `R` = ops per qubit per layer is a config field, `operations_per_qubit` (3 for VQC's
  RZ/RY/RX). `RotationShape(L, N, R)` is built from that plus the layer/qubit counts.
- **Named aux weights** — `aux_defaults: dict[str, float]` per circuit, overridable from Hydra
  config by name (`cfg.aux_weights.scale_factor`, etc.), instead of a magic slice. VQC now says
  `weights.aux['scale_factor']` instead of `weights[-1]`. `hamiltonian_coeffs` (VQC's Hamiltonian,
  today's fixed `weights[-11:-1]`, 10 coefficients regardless of qubit count) has no static default
  since its correct length depends on qubit count at runtime: use
  `cfg.aux_weights.hamiltonian_coeffs` if present (validated against `len(wires)`), else default to
  `[0.1] * len(wires)`. Checked against the run archive: every past run using this Hamiltonian
  measurement used exactly 10 qubits, so this changes no existing run's behavior — only what
  happens at other qubit counts (now a clear error instead of a silent mismatch).

`weights.aux['bias']` (today's `sum(weights[-6:-1])`) is read in `losses.py`, not inside the
circuit — same as today, just a named lookup instead of a slice (D9, below).

D6/D7/D8 (all done, see `refactored.MD`) confirmed the rest of A3 works end to end: a
`CircuitWeights` instance derives its QFI state-circuit generically, matches today's hardcoded
methods bit-for-bit, and round-trips losslessly through the existing checkpoint mechanism —
including PennyLane's optimizer accepting `rot` plus several named aux scalars as separate
trainable arguments in one `step_and_cost` call.

---

## Part B — Remaining deliverables

D1, D2, D4, D5, D6, D7, D8, D9, D11 are done (`refactored.MD`). D10 needs D9 (done — unblocked).
D12 is a proposed addendum, independent of D10, not yet decided.

```
D10 (e2e run)
D12 (QFI/CircuitWeights compatibility, proposed, independent)
```

### D10 — End-to-end validation run

**Depends on:** D9. **Files:** none (a training run, not a code change).

Re-run one short training job (a handful of epochs, small config) end to end.

**Done when:** the loss curve and checkpoint format match a pre-refactor run of the same config
(checkpoint content adjusted for `CircuitWeights` serialization, per D7).

### D12 — QFI/CircuitWeights compatibility *(proposed, not yet decided)*

**Depends on:** nothing. **Files:** `quantum/architectures.py` (`quantum_fisher`,
`run_fisher_computation`).

**New finding (surfaced while verifying D9):** these two methods still assume a flat weight
vector — `quantum_fisher` slices `full_qfi[:3*N*self.num_layers, :3*N*self.num_layers]` and hands
`self.current_weights` to `qml.metric_tensor`/`qml.adjoint_metric_tensor` as one differentiable
argument. Confirmed broken by direct test: calling `quantum_fisher` with a real `CircuitWeights`
raises `AttributeError: 'ArrayBox' object has no attribute 'item'` inside PennyLane's own
differentiation machinery. Out of D9's declared scope (not listed in its file list), so left
unfixed; fixing it needs the same multi-arg differentiation pattern D7 confirmed for the optimizer
(`rot` plus named aux values passed as separate args to `metric_tensor`), and the `3*N*num_layers`
slice replaced with a direct `rotation_shape`-derived size.

**Also found, not fixed (pre-existing, unrelated to D9):** `QuantumTrainer.iteration()`'s
validation branch (`train=False`) does `float(scores)`, which only works when `scores` is a single
value — i.e. `batch_size=1`. Confirmed this line is untouched by any D9 change. Matches
`evaluate.py`'s own documented constraint ("`run_inference` expects `batch_size=1`"), but
`run_training_loop`'s validation phase calls the same code path with `cfg.batch_size` (100 by
default in `VQC/base.yaml`) — would already have failed there before this refactor. Not part of
D12; noted here since it surfaced during the same verification pass.

**Done when:** `quantum_fisher`/`run_fisher_computation` run against a `CircuitWeights` instance
without error and reproduce the pre-refactor QFI matrix for an equivalent flat-weight input.
