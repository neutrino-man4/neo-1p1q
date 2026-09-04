# Circuit Module Refactor

Companion to `AUDIT.md`, section 3.2 ("modular, plug-and-play circuit design"): target interface,
both existing circuits migrated into it, quirks found during migration, and a non-breaking,
step-by-step path to get there.

## 1. Goal

`quantum/architectures.py` hardcodes two circuits (`_vqc_circuit` / `_qcnn_implementation`) as
private methods of `QuantumClassifier`, selected by a string switch in `set_circuit()`, each with a
hand-duplicated `expval` version and `state()` version (for QFI). Adding a circuit means editing
`QuantumClassifier` in four places and extending the switch.

Target: a new circuit is a new file implementing a small `Circuit` protocol, registered by name.
`QuantumClassifier` and `QuantumTrainer` do not change when a circuit is added.

## 2. Four operations, one non-universal loop

| | VQC (`_vqc_circuit`) | QCNN (`_qcnn_implementation`) |
|---|---|---|
| **encode** | every layer, full wire set, offset into `inputs` by `w + L*N` | once, before any layers, offset by `w` only |
| **entangle** | fixed `CNOT` ring, all N wires, every layer | one initial `CY` ring, then folded into the per-layer loop |
| **rotate** | one weight triplet (RZ/RY/RX) per wire, after entangle | fused with entangle: `RZ`/`RY` on each surviving wire-pair then `CNOT`, one shared (phi, theta) pair per layer |
| **wires per layer** | all N, every layer | shrinking set `auto_wires[:-(1+L)]` (pooling) |
| **measure** | scalar `expval` of a weighted `Hamiltonian` over all wires | list of `expval(PauliZ(i))`, one per wire |

`encode`, `entangle`, `rotate`, `measure` are the right unit of reuse; the **composition** of the
four (the loop shape) is not — QCNN's pooling topology can't be expressed by one master
"per-layer: encode, entangle, rotate" loop. So the loop (`build()`) must stay overridable, with a
default covering the common case.

## 3. Weight representation: a derived rotation tensor + named aux weights

Today's flat weight vector mixes two unrelated things via magic negative indices: per-qubit
rotation angles (`weights[start:start+3*N:3]`-style slicing) and auxiliary scalars — scale factor,
loss bias, Hamiltonian coefficients — stuffed into the tail (`weights[-1]`, `weights[-2]`,
`weights[-6:-1]`, `weights[-11:-1]`, inconsistent between circuits, one of them hardcoded a second
time in `losses.py`). This refactor replaces both with two explicit pieces.

**Rotation tensor**, shape `(L, N, R)`:
- `L` = `cfg.num_layers`, `N` = qubit count (`len(auto_wires)`) — both already exist in config.
- `R` = ops per qubit per layer — the **one new field**, `operations_per_qubit`, added to config
  next to `wires`/`num_layers`. A circuit never hand-assembles this tuple; it declares
  `operations_per_qubit` (3 for VQC's RZ/RY/RX, 2 for QCNN's phi/theta) and a shared default method
  builds `RotationShape(L, N, R)` from it plus the existing `num_layers`/qubit-count. The only
  override needed is when a circuit's rotation isn't per-qubit at all: QCNN's (phi, theta) is one
  shared pair per layer, broadcast to every active wire, so it overrides just `N → 1`.

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
array out-of-bounds error, so no separate check is needed there. `validate()` runs once per layer
inside `build()`, so a config requesting more layers/wires than the run was sized for fails with a
named reason instead of a shape error deep inside PennyLane.

**Named aux weights** — declared per circuit as `aux_defaults: dict[str, float]`, each overridable
from Hydra config by name:

```yaml
# hydra_configs/VQC/base.yaml
wires: 4
num_layers: 1
operations_per_qubit: 3        # new -- see above
aux_weights:
  scale_factor: 1.0
  bias: 0.1
```

`train.py` merges `{**circuit.aux_defaults, **cfg.get('aux_weights', {})}`. Both circuits now say
`weights.aux['scale_factor']` — no more disagreement between `weights[-1]` and `weights[-2]`, which
existed only because their neighboring slices claimed different amounts of trailing space.
`hamiltonian_coeffs` (VQC's Hamiltonian, today's fixed `weights[-11:-1]`, 10 coefficients
regardless of qubit count) is the one aux entry whose correct length depends on a runtime value, so
it has no static default: `VQCCircuit.measure()` uses `cfg.aux_weights.hamiltonian_coeffs` if
present (validating `len(...) == len(wires)`, raising otherwise), else defaults to
`[0.1] * len(wires)`. Checked against the run archive: every past run using this Hamiltonian
measurement used exactly 10 qubits, so this changes no existing run's behavior — only what happens
at other qubit counts (now a clear error instead of a silent mismatch).

`weights.aux['bias']` (today's `sum(weights[-6:-1])`) is read in `losses.py`, not inside the
circuit — same as today, just a named lookup; updating that call site is a required knock-on change
(section 8), outside `quantum/circuits/` itself.

**Unrelated note from the same migration**: `QuantumClassifier.__init__`/`_initialize_wires` sets
`use_ancilla`, `truth_wire`, `separate_ancilla`, none of which either circuit reads — likely
vestigial from an earlier ancilla-based measurement scheme. Decide during migration whether to
carry them into the new protocol or drop them.

**Open risk — confirm before implementing, not just designing**: `QuantumTrainer` currently calls
`qml.AdamOptimizer.step_and_cost(cost_fn, weights)` on one flat trainable array. Splitting weights
into `rot` (array) + named `aux` scalars means the optimizer step must track several trainable
objects (e.g. `opt.step_and_cost(cost_fn, rot, *aux.values())`, repacked into `CircuitWeights`
after), and checkpoint save/load (currently a flat-array pickle) must serialize the same structure.
This touches `QuantumTrainer`, not just `quantum/circuits/` — confirm PennyLane's optimizer
supports multiple trainable args before committing to this shape.

## 4. The `Circuit` protocol

```python
# quantum/circuits/base.py
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
        Override only when N isn't the qubit count (e.g. QCNN's shared per-layer params)."""
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

`CircuitBase` implements `rotation_shape()` and `build()` exactly as shown, so a standard circuit
subclasses it and only writes `operations_per_qubit`, `aux_defaults`, `encode`, `entangle`,
`rotate`, `measure` — no shape tuple, no loop. QCNN subclasses `Circuit` directly (or `CircuitBase`
with `rotation_shape`/`build` overridden) because its loop shape and its `N` genuinely differ.

## 5. VQC migrated (fits the default `build()`/`rotation_shape()` unchanged)

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

## 6. QCNN migrated (owns `rotation_shape()` and `build()` — the loop shape genuinely differs)

```python
# quantum/circuits/qcnn.py
import pennylane as qml
import pennylane.numpy as np
from .base import Circuit, RotationShape
from helpers.utils import getIndex

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class QCNNCircuit(Circuit):
    operations_per_qubit = 2   # phi, theta -- shared across active wires, not per-qubit
    aux_defaults = {'scale_factor': 1.0}

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'),
            'pt':  getIndex('particle', 'pt'),
        }

    def rotation_shape(self, n_qubits, n_layers):
        return RotationShape(L=n_layers, N=1, R=self.operations_per_qubit)

    def encode(self, weights, inputs, layer, wires):
        sf = 2 * np.pi * sigmoid(weights.aux['scale_factor']) + 1
        for w in wires:
            zenith  = np.squeeze(inputs[:, w, self.index['eta']])
            azimuth = np.squeeze(inputs[:, w, self.index['phi']])
            radius  = np.squeeze(inputs[:, w, self.index['pt']])
            if inputs.shape[0] == 1:
                zenith, azimuth, radius = zenith.item(), azimuth.item(), radius.item()
            qml.RY(sf * radius * zenith, wires=w)
            qml.RZ(sf * radius * azimuth, wires=w)

    def entangle(self, wires):
        N = len(wires)
        for w in wires:
            qml.CY(wires=[w, (w + 1) % N])

    def rotate(self, weights, layer, wires):
        # "wires" is the pooled/active set for this layer -- see build() below
        N = len(wires) + layer + 1   # original modulus base before pooling
        phi, theta = weights.rot[layer, 0, 0], weights.rot[layer, 0, 1]
        for w in wires:
            qml.RZ(phi, wires=w)
            qml.RY(theta, wires=w)
            qml.RZ(phi, wires=(w + layer + 1) % N)
            qml.RY(theta, wires=(w + layer + 1) % N)
            qml.CNOT(wires=[w, (w + 1 + layer) % N])

    def measure(self, weights, wires):
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    def build(self, weights, inputs, wires):
        shape = self.rotation_shape(len(wires), self.num_layers)
        self.encode(weights, inputs, layer=0, wires=wires)   # once, not per layer
        self.entangle(wires)                                 # initial ring only
        active = list(wires)
        for L in range(self.num_layers):
            shape.validate(L, active)
            active = active[:-(1 + L)]                       # pooling: one fewer wire per layer
            self.rotate(weights, layer=L, wires=active)
        return self.measure(weights, wires)                  # measured over ALL original wires
```

## 7. State-circuit derivation for QFI

`quantum_fisher()`/`run_fisher_computation()` need a version of the circuit returning `qml.state()`
instead of `measure()`'s expval — today a hand-copied second method per circuit. Since `build()`
always ends by calling `self.measure(...)`, derive the state version generically: give
`Circuit.build` an optional `measure_override: Callable | None = None`, called instead of
`self.measure(...)` when set:

```python
# quantum/architectures.py (QuantumClassifier)
def _state_build(self, weights: CircuitWeights, inputs, wires):
    return self._impl.build(weights, inputs, wires, measure_override=lambda *_: qml.state())
```

One circuit implementation now produces both the expval and state versions — a new circuit gets
QFI support for free as long as its `build()` calls `self.measure(...)` exactly once, at the end.

Note: `set_circuit()` currently wraps only the VQC state circuit in `qml.defer_measurements(...)`,
not the QCNN one — worth checking whether that's load-bearing (neither circuit appears to use
mid-circuit measurement today) before deciding whether the generic wrapper needs it too.

## 8. Registry and `QuantumClassifier` integration

```python
# quantum/circuits/registry.py
from .vqc import VQCCircuit
from .qcnn import QCNNCircuit

_REGISTRY = {'normal': VQCCircuit, 'CNN': QCNNCircuit}

def get(circuit_type: str, num_layers: int) -> Circuit:
    try:
        return _REGISTRY[circuit_type](num_layers=num_layers)
    except KeyError:
        raise ValueError(f"Unknown circuit_type '{circuit_type}'. Registered: {list(_REGISTRY)}")
```

`QuantumClassifier.set_circuit()` collapses to:

```python
def set_circuit(self, circuit_type: str = 'normal') -> None:
    self._impl = circuits.registry.get(circuit_type, self.num_layers)
    self.circuit = qml.QNode(self._impl.build, self.device, interface=self.backend)
    self.state_circuit_qnode = qml.QNode(self._state_build, self.device, interface=self.backend)
```

`cfg.operations_per_qubit` overrides the circuit's structural default when set — pass it through
after construction: `VQC._impl.operations_per_qubit = cfg.get('operations_per_qubit', VQC._impl.operations_per_qubit)`.

`train.py`'s weight construction (today: `NUM_WEIGHTS = ... + cfg.extra_weights`, a flat
`np.random.uniform` array with `init_weights[-cfg.extra_weights:-1] = 0.1`) becomes:

```python
shape = VQC._impl.rotation_shape(len(VQC.auto_wires), cfg.num_layers)
rot = np.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R))
aux = {**VQC._impl.aux_defaults, **dict(cfg.get('aux_weights', {}))}
init_weights = CircuitWeights(rot=rot, aux=aux)
```

`cfg.extra_weights` and its `train.py:109-121` branch are dropped entirely — every trainable
quantity is now either part of the shaped rotation tensor or a named aux entry; there's no leftover
count to reconcile.

`losses.py`'s `VQC_cost`/`probabilistic_loss` read `weights.aux['bias']` instead of
`sum(weights[-6:-1])` — a required `losses.py` signature change, a direct consequence of removing
the flat vector, outside `quantum/circuits/`'s own scope.

## 9. Migration plan (incremental, non-breaking)

1. Add `quantum/circuits/base.py` (`RotationShape`, `CircuitWeights`, `Circuit` protocol,
   `CircuitBase`'s default `rotation_shape()`/`build()`). No existing code changes yet.
2. Add `quantum/circuits/vqc.py` and `quantum/circuits/qcnn.py` (sections 5-6).
3. Add `quantum/circuits/registry.py`.
4. Add `operations_per_qubit` and `aux_weights:` to `hydra_configs/{VQC,QAE,VPC,QFI,AOJ}/base.yaml`.
5. Confirm the optimizer/checkpoint risk (section 3) — `QuantumTrainer`'s `step_and_cost` and
   checkpoint (de)serialization must handle `rot` + named `aux` as separate trainables, not one
   flat array — before relying on this shape.
6. Parity test: fixed seed, a `CircuitWeights` built to match today's flat weight values exactly,
   fixed input batch; assert `VQCCircuit().build(...)` / `QCNNCircuit().build(...)` match today's
   `_vqc_circuit`/`_qcnn_implementation` output (float tolerance). This is the actual safety net.
7. Only once step 6 passes: switch `set_circuit()` to the registry version, delete
   `_vqc_circuit`/`_vqc_state_circuit`/`_qcnn_implementation`/`_qcnn_state_circuit`, switch
   `train.py`'s weight construction, update `losses.py`'s bias read.
8. Re-run one short training job end to end; confirm the loss curve and checkpoint format are
   unchanged (adjusted for `CircuitWeights` serialization) before relying on this for real runs.

Steps 1-4 touch no existing behavior and can be merged on their own. Step 7 is the only point where
old code paths are removed — gate it on the parity test, not on inspection.
