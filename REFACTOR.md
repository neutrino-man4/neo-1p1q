# Circuit Module Refactor

Companion to `AUDIT.md`, section 3.2 ("modular, plug-and-play circuit design"). Part A is
background for the design. Part B covers the remaining deliverables, ordered by dependency.

**Status:** D1, D2, D4, D5, D6, D7, D8 are implemented — see `refactored.MD`. This file covers
what's left: D9, D10, and D11 (proposed, not yet decided).

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

D1, D2, D4, D5, D6, D7, D8 are done (`refactored.MD`). D9 is now unblocked — every dependency is
complete. D10 needs D9. D11 is a proposed addendum, independent of the others, not yet decided.

```
D9 (switchover) ─▶ D10 (e2e run)
D11 (loss registry, proposed/optional, independent)
```

### D9 — Switchover

**Depends on:** D6, D7, D8 (all done). **Files:** `quantum/architectures.py`, `train.py`,
`quantum/losses.py`.

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
5. **New finding (surfaced while implementing D7):** `QuantumTrainer.iteration()` still calls
   `self.optim.step_and_cost(self.quantum_loss, self.current_weights, ...)` with one flat trainable
   argument. This must become the multi-arg form confirmed working in D7 —
   `step_and_cost(cost_fn, rot, *aux.values())` — with the returned tuple repacked into a
   `CircuitWeights` after every step. This wasn't listed in the original D9 scope; it depends on
   step 4 above (`losses.py` must accept `CircuitWeights`-shaped args first).

**Done when:** D8's parity test still passes with the registry-based circuit wired in for real
(not just called standalone), and no references to the deleted methods, `circuit_type='CNN'`, or
`cfg.extra_weights` remain.

### D10 — End-to-end validation run

**Depends on:** D9. **Files:** none (a training run, not a code change).

Re-run one short training job (a handful of epochs, small config) end to end.

**Done when:** the loss curve and checkpoint format match a pre-refactor run of the same config
(checkpoint content adjusted for `CircuitWeights` serialization, per D7).

### D11 — Loss function registry *(proposed, not yet decided: fold into D9 or keep standalone)*

**Depends on:** nothing. **Files:** `quantum/losses.py`.

**New finding (surfaced while reviewing `losses.py` for D9):** `VQC_cost` and `batched_VQC_cost`
each hardcode the same `if loss_type=='BCE': ... elif loss_type=='MSE': ... else: sys.exit(-1)` —
duplicated verbatim in both functions, with an unrecoverable `sys.exit(-1)` on an unknown type
(inconsistent with the `ValueError` style already used by `circuits.registry.get()`).
`probabilistic_loss` takes a `loss_type` parameter but never branches on it at all — a dead
parameter, noted here, not touched.

Same fix shape as D4's circuit registry: a single `LOSS_FNS: Dict[str, Callable]` mapping name to
scoring function (e.g. `{'MSE': mfunc.mean_squared_error, 'BCE': mfunc.binary_cross_entropy}`),
looked up once per call, raising on an unknown name instead of exiting the process. Adding a new
loss becomes "write the function, add one dict entry" — no existing cost function edited.

**Done when:** `VQC_cost`, `batched_VQC_cost`, and `probabilistic_loss` all resolve their scoring
function through one shared lookup; an unknown `loss_type` raises `ValueError` (listing valid
names) instead of calling `sys.exit`.
