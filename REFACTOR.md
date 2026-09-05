# Circuit Module Refactor

Companion to `AUDIT.md`, section 3.2 ("modular, plug-and-play circuit design"). Part A is
background for the design. Part B covers the remaining deliverables, ordered by dependency.

**Status:** D1, D2, D4, D5 are implemented — see `refactored.MD`. This file covers what's left:
D6, D7, D8, D9, D10.

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
  RZ/RY/RX). `RotationShape(L, N, R)` is built from that plus the layer/qubit counts (see
  `refactored.MD` D1 for `RotationShape`).
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

---

## Part B — Remaining deliverables

D1, D2, D4, D5 are done (`refactored.MD`). D6 and D8 are now unblocked (their only dependency, D4,
is complete). D7 is independent and can proceed now. D9 needs D6, D7, D8 all complete; D10 needs D9.

```
D6 (QFI state) ─┐
D8 (parity test)─┼─▶ D9 (switchover) ─▶ D10 (e2e run)
D7 (trainer: optimizer + checkpoint, independent) ─┘
```

### D6 — QFI state-circuit derivation

**Depends on:** D4 (done). **Files:** `quantum/architectures.py` (additive: one new method, no
deletions yet — those happen in D9).

`quantum_fisher()`/`run_fisher_computation()` need a version of the circuit returning `qml.state()`
instead of `measure()`'s expval — today a hand-copied second method (`_vqc_state_circuit`). Since
`build()` always ends by calling `self.measure(...)`, derive the state version generically: give
`Circuit.build` an optional `measure_override: Callable | None = None`, called instead of
`self.measure(...)` when set:

```python
def _state_build(self, weights: CircuitWeights, inputs, wires):
    return self._impl.build(weights, inputs, wires, measure_override=lambda *_: qml.state())
```

**Done when:** `_state_build` produces the same state vector as today's `_vqc_state_circuit` for a
fixed seed/weights/input (compare directly, this is a small parity check of its own).

### D7 — Trainer support for split weights *(independent spike, gates D9)*

**Depends on:** nothing. **Files:** `quantum/architectures.py` (`QuantumTrainer`).

`QuantumTrainer` currently calls `qml.AdamOptimizer.step_and_cost(cost_fn, weights)` on one flat
trainable array, and checkpoints that array via pickle. Splitting weights into `rot` (array) +
named `aux` scalars means the optimizer step must track several trainable objects, e.g.
`opt.step_and_cost(cost_fn, rot, *aux.values())`, repacked into `CircuitWeights` after each step —
and checkpoint save/load must serialize the same structure.

**Done when:** a throwaway script confirms `qml.AdamOptimizer.step_and_cost` accepts and updates
multiple trainable args as above, and `QuantumTrainer`'s checkpoint read/write round-trips a
`CircuitWeights` instance losslessly. This is a prerequisite, not optional — if PennyLane's
optimizer doesn't support this cleanly, the weight representation in A3 needs revisiting before D9.

### D8 — Parity test

**Depends on:** D2, D4 (both done). **Files:** new test module (e.g. `tests/test_circuit_parity.py`).

For a fixed seed, a `CircuitWeights` built to match today's flat weight values exactly, and a fixed
input batch: assert `VQCCircuit().build(...)` matches today's `_vqc_circuit` output to float
tolerance.

**Done when:** the test passes against the current (pre-switchover) `architectures.py`. This is the
actual safety net for D9 — gate D9 on this test, not on inspection.

### D9 — Switchover

**Depends on:** D6, D7, D8 all complete (D5 is already done). **Files:** `quantum/architectures.py`,
`train.py`, `quantum/losses.py`.

1. `QuantumClassifier.set_circuit()`:
   ```python
   def set_circuit(self, circuit_type: str = 'normal') -> None:
       self._impl = circuits.registry.get(circuit_type, self.num_layers)
       self._impl.operations_per_qubit = cfg.get('operations_per_qubit', self._impl.operations_per_qubit)
       self.circuit = qml.QNode(self._impl.build, self.device, interface=self.backend)
       self.state_circuit_qnode = qml.QNode(self._state_build, self.device, interface=self.backend)
   ```
2. Delete `_vqc_circuit`, `_vqc_state_circuit`, `_qcnn_implementation`, `_qcnn_state_circuit` — all
   four, not just the VQC pair. QCNN is out of scope for this version (see the note at the top of
   this document).
3. `train.py` weight construction — replace `NUM_WEIGHTS = ... + cfg.extra_weights` and
   `init_weights[-cfg.extra_weights:-1] = 0.1` with:
   ```python
   shape = VQC._impl.rotation_shape(len(VQC.auto_wires), cfg.num_layers)
   rot = np.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R))
   aux = {**VQC._impl.aux_defaults, **dict(cfg.get('aux_weights', {}))}
   init_weights = CircuitWeights(rot=rot, aux=aux)
   ```
   Drop `cfg.extra_weights` and `train.py:109-121` entirely.
4. `quantum/losses.py`: `VQC_cost`/`probabilistic_loss` read `weights.aux['bias']` instead of
   `sum(weights[-6:-1])`.

**Done when:** D8's parity test still passes with the registry-based circuit wired in for real
(not just called standalone), and no references to the deleted methods, `circuit_type='CNN'`, or
`cfg.extra_weights` remain.

### D10 — End-to-end validation run

**Depends on:** D9. **Files:** none (a training run, not a code change).

Re-run one short training job (a handful of epochs, small config) end to end.

**Done when:** the loss curve and checkpoint format match a pre-refactor run of the same config
(checkpoint content adjusted for `CircuitWeights` serialization, per D7).
