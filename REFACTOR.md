# Circuit Module Refactor

Companion to `AUDIT.md`, section 3.2 ("modular, plug-and-play circuit design"). Part A is
background: the target design. Part B breaks the work into standalone deliverables, ordered by
dependency, each small enough to implement and merge on its own.

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
`weights[-11:-1]`, one of them hardcoded a second time in `losses.py`). Replace both with two
explicit pieces:

- **Rotation tensor**, shape `(L, N, R)` — `L = cfg.num_layers`, `N` = qubit count, both already in
  config. `R` = ops per qubit per layer is the **one new config field**, `operations_per_qubit` (3
  for VQC's RZ/RY/RX). A circuit never hand-assembles this tuple: it declares
  `operations_per_qubit`, and a shared default builds `RotationShape(L, N, R)` from that plus the
  existing layer/qubit counts.
- **Named aux weights** — `aux_defaults: dict[str, float]` per circuit, overridable from Hydra
  config by name (`cfg.aux_weights.scale_factor`, etc.), instead of a magic slice. VQC now says
  `weights.aux['scale_factor']` instead of `weights[-1]`. `hamiltonian_coeffs` (VQC's Hamiltonian,
  today's fixed `weights[-11:-1]`, 10 coefficients regardless of qubit count) has no static default
  since its correct length depends on qubit count at runtime: use
  `cfg.aux_weights.hamiltonian_coeffs` if present (validated against `len(wires)`), else default to
  `[0.1] * len(wires)`. Checked against the run archive: every past run using this Hamiltonian
  measurement used exactly 10 qubits, so this changes no existing run's behavior — only what
  happens at other qubit counts (now a clear error instead of a silent mismatch).

```python
@dataclass(frozen=True)
class RotationShape:
    L: int
    N: int
    R: int

    def validate(self, layer: int, wires: list[int]) -> None:
        if layer >= self.L:
            raise ValueError(f"layer {layer} exceeds declared L={self.L}")
        if self.N > 1 and len(wires) > self.N:
            raise ValueError(f"{len(wires)} wires requested, declared N={self.N}")
```

`rotate()` indexes the tensor directly (`weights.rot[layer, :, op]`); an op index `>= R` is a plain
out-of-bounds error, so no separate check is needed there. `validate()` runs once per layer inside
`build()`, so a config requesting more layers/wires than declared fails with a named reason instead
of a shape error inside PennyLane.

`weights.aux['bias']` (today's `sum(weights[-6:-1])`) is read in `losses.py`, not inside the
circuit — same as today, just a named lookup instead of a slice.

### A4. Notes carried forward, not yet decided

- `QuantumClassifier.__init__`/`_initialize_wires` sets `use_ancilla`, `truth_wire`,
  `separate_ancilla`, none of which `_vqc_circuit` reads — likely vestigial from an earlier
  ancilla-based measurement scheme. Decide in D2 (below) whether to carry them into the new
  protocol or drop them.

---

## Part B — Deliverables

Each deliverable is independently mergeable once its dependencies land. D1-D6 change no existing
behavior. D7 is a research/implementation spike gating D9. D8 is the parity safety net gating D9.
D9 is the only deliverable that removes old code paths (including QCNN's).

```
D1 (base types) ──▶ D2 (VQC) ──▶ D4 (registry) ─┬─▶ D6 (QFI state) ─┐
                                                  ├─▶ D8 (parity test)─┤
D5 (Hydra config, independent) ───────────────────┘                   ▼
D7 (trainer: optimizer + checkpoint, independent) ─────────────▶ D9 (switchover) ─▶ D10 (e2e run)
```

### D1 — Weight types and `Circuit` protocol

**Depends on:** nothing. **Files:** new `quantum/circuits/base.py`.

```python
from dataclasses import dataclass
from typing import Protocol, List, Dict, Any
import pennylane as qml
import pennylane.numpy as np

@dataclass
class CircuitWeights:
    rot: np.ndarray          # shape (L, N, R)
    aux: Dict[str, Any]      # name -> value, circuit.aux_defaults merged with cfg.aux_weights

class Circuit(Protocol):
    num_layers: int
    operations_per_qubit: int      # R -- the only shape field a circuit declares
    aux_defaults: Dict[str, float]

    def rotation_shape(self, n_qubits: int, n_layers: int) -> RotationShape:
        """Default: RotationShape(L=n_layers, N=n_qubits, R=self.operations_per_qubit).
        Override only if a future circuit's rotation isn't per-qubit (N != qubit count)."""
        return RotationShape(L=n_layers, N=n_qubits, R=self.operations_per_qubit)

    def encode(self, weights: CircuitWeights, inputs: np.ndarray, layer: int, wires: List[int]) -> None:
        """State preparation from data. Takes `weights` so a circuit can read a
        named aux value (e.g. scale_factor) with no weights-free/weights-taking ambiguity."""
        ...

    def entangle(self, wires: List[int]) -> None:
        """Fixed (non-parameterized) entangling gates over the given wires."""
        ...

    def rotate(self, weights: CircuitWeights, layer: int, wires: List[int]) -> None:
        """Trainable single-qubit rotations for this layer's active wires."""
        ...

    def measure(self, weights: CircuitWeights, wires: List[int]):
        """Terminal measurement: an expval (scalar or Hamiltonian), or a list of expvals."""
        ...

    def build(self, weights: CircuitWeights, inputs: np.ndarray, wires: List[int]):
        """Full circuit body; QNode is built directly from this method. Default
        composition below; override only when the loop shape itself differs."""
        shape = self.rotation_shape(len(wires), self.num_layers)
        for L in range(self.num_layers):
            shape.validate(L, wires)
            self.encode(weights, inputs, layer=L, wires=wires)
            self.entangle(wires)
            self.rotate(weights, layer=L, wires=wires)
        return self.measure(weights, wires)
```

`CircuitBase` implements `rotation_shape()` and `build()` exactly as shown.

**Done when:** `base.py` imports cleanly; `RotationShape.validate` and `CircuitBase.build`/
`rotation_shape` have unit tests covering the layer/wire-overrun error paths. No other file
changes.

### D2 — VQC circuit

**Depends on:** D1. **Files:** new `quantum/circuits/vqc.py`.

```python
# quantum/circuits/vqc.py
import pennylane as qml
import pennylane.numpy as np
from .base import CircuitBase, CircuitWeights
from helpers.utils import getIndex

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class VQCCircuit(CircuitBase):
    operations_per_qubit = 3   # RZ, RY, RX per wire per layer
    aux_defaults = {'scale_factor': 1.0, 'bias': 0.1}
    # 'hamiltonian_coeffs' has no static default -- its length depends on
    # qubit count, resolved in measure() below.

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'),
            'pt':  getIndex('particle', 'pt'),
        }

    def encode(self, weights, inputs, layer, wires):
        N = len(wires)
        sf = 2 * np.pi * sigmoid(weights.aux['scale_factor']) + 1
        for w in wires:
            zenith  = np.squeeze(inputs[:, w + layer * N, self.index['eta']])
            azimuth = np.squeeze(inputs[:, w + layer * N, self.index['phi']])
            radius  = np.squeeze(inputs[:, w + layer * N, self.index['pt']])
            if inputs.shape[0] == 1:
                zenith, azimuth, radius = zenith.item(), azimuth.item(), radius.item()
            qml.RY(sf * radius * zenith, wires=w)
            qml.RX(sf * radius * azimuth, wires=w)

    def entangle(self, wires):
        N = len(wires)
        for w in wires:
            qml.CNOT(wires=[w, (w + 1) % N])

    def rotate(self, weights, layer, wires):
        rot_z, rot_y, rot_x = (weights.rot[layer, :, i] for i in range(3))
        for rz, ry, rx, w in zip(rot_z, rot_y, rot_x, wires):
            qml.Rot(0., ry, rz, wires=w)
            qml.RX(rx, wires=w)

    def measure(self, weights, wires):
        obs = [qml.PauliZ(i) for i in wires]
        coeffs = weights.aux.get('hamiltonian_coeffs', [0.1] * len(wires))
        if len(coeffs) != len(wires):
            raise ValueError(
                f"hamiltonian_coeffs has {len(coeffs)} entries, circuit has {len(wires)} wires"
            )
        return qml.expval(qml.Hamiltonian(coeffs, obs))
```

Resolve the ancilla-attributes question from A4 here (carry into `Circuit` or drop).

**Done when:** `VQCCircuit().build(...)` runs standalone (fixed dummy weights/inputs, no
`QuantumClassifier` involved) and returns a scalar expval.

### D4 — Registry

**Depends on:** D2. **Files:** new `quantum/circuits/registry.py`.

```python
from .vqc import VQCCircuit

_REGISTRY = {'normal': VQCCircuit}

def get(circuit_type: str, num_layers: int) -> Circuit:
    try:
        return _REGISTRY[circuit_type](num_layers=num_layers)
    except KeyError:
        raise ValueError(f"Unknown circuit_type '{circuit_type}'. Registered: {list(_REGISTRY)}")
```

Only `'normal'` is registered — `circuit_type='CNN'` now raises `ValueError` rather than
dispatching anywhere; that's expected until QCNN is reintroduced as its own deliverable.

**Done when:** `registry.get('normal', 1)` returns a `VQCCircuit` instance; any other name raises
`ValueError` listing the registered names.

### D5 — Hydra config fields

**Depends on:** nothing (can be done any time before D9). **Files:**
`hydra_configs/{VQC,QAE,VPC,QFI,AOJ}/base.yaml`.

Add, next to the existing `wires`/`num_layers`:

```yaml
operations_per_qubit: 3        # VQC's RZ/RY/RX per wire per layer
aux_weights:
  scale_factor: 1.0
  bias: 0.1
```

Any config using `circuit_type: 'CNN'` needs to be flagged or updated separately — it has no
circuit to dispatch to after D9 (see D4).

**Done when:** every `base.yaml` under those five projects declares both fields with a value
matching `VQCCircuit.operations_per_qubit`.

### D6 — QFI state-circuit derivation

**Depends on:** D4. **Files:** `quantum/architectures.py` (additive: one new method, no deletions
yet — those happen in D9).

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

**Depends on:** nothing (can start in parallel with D1-D6). **Files:** `quantum/architectures.py`
(`QuantumTrainer`).

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

**Depends on:** D2, D4. **Files:** new test module (e.g. `tests/test_circuit_parity.py`).

For a fixed seed, a `CircuitWeights` built to match today's flat weight values exactly, and a fixed
input batch: assert `VQCCircuit().build(...)` matches today's `_vqc_circuit` output to float
tolerance.

**Done when:** the test passes against the current (pre-switchover) `architectures.py`. This is the
actual safety net for D9 — gate D9 on this test, not on inspection.

### D9 — Switchover

**Depends on:** D5, D6, D7, D8 all complete. **Files:** `quantum/architectures.py`, `train.py`,
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

**Done when:** D8's parity test still passes with the registry-based circuit wired in for real
(not just called standalone), and no references to the deleted methods, `circuit_type='CNN'`, or
`cfg.extra_weights` remain.

### D10 — End-to-end validation run

**Depends on:** D9. **Files:** none (a training run, not a code change).

Re-run one short training job (a handful of epochs, small config) end to end.

**Done when:** the loss curve and checkpoint format match a pre-refactor run of the same config
(checkpoint content adjusted for `CircuitWeights` serialization, per D7).
