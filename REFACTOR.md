# Circuit Module Refactor

Companion to `AUDIT.md`. Covers section 3.2 of that document ("modular, plug-and-play circuit
design") in implementation detail: the target interface, both existing circuits migrated into it,
the quirks uncovered while doing that migration, and a step-by-step, non-breaking path to get
there.

---

## 1. Goal

Today `quantum/architectures.py` hardcodes two circuits (`_vqc_circuit` / `_qcnn_implementation`)
as private methods of `QuantumClassifier`, selected by a string switch in `set_circuit()`, each
with a hand-duplicated `expval` version and `state()` version (for QFI). Adding a circuit means
editing `QuantumClassifier` in four places and extending the switch.

Target: a new circuit is a new file implementing a small `Circuit` protocol, registered by name.
`QuantumClassifier` and `QuantumTrainer` do not change when a circuit is added.

## 2. The four operations are real, but the loop that composes them is not universal

Diffing the two existing circuits operation-by-operation:

| | VQC (`_vqc_circuit`) | QCNN (`_qcnn_implementation`) |
|---|---|---|
| **encode** | re-run every layer, full wire set, offset into `inputs` by `w + L*N` | run once, before any layers, offset by `w` only |
| **entangle** | fixed ring `CNOT` over all N wires, identical every layer | one initial ring of `CY`, then folded into the per-layer loop below |
| **rotate** | separate loop over all wires, after entangle, one weight triplet (RZ/RY/RX) per wire | fused with entangle: per surviving wire-pair, `RZ`/`RY` on both members then one `CNOT`, one shared (phi, theta) pair per layer |
| **wires touched per layer** | all `N` wires, every layer | shrinking set `auto_wires[:-(1+L)]` — one fewer wire each layer (pooling) |
| **measure** | one scalar: `expval` of a weighted `Hamiltonian` over all wires | a list: `expval(PauliZ(i))` per wire |

QCNN's rotate and entangle are not sequential per-layer steps over a fixed wire set — they're
interleaved per shrinking wire-pair, and its encoding isn't per-layer at all. So:

- The four operations (`encode`, `entangle`, `rotate`, `measure`) are the right *unit* of reuse.
- The **composition** of those four — the loop shape — is not; it must stay overridable per
  circuit, with a default implementation covering the common case (repeat encode → entangle →
  rotate every layer over a fixed wire set, measure once).

## 3. Weight representation: rotation tensor + named aux weights

Today's flat weight vector mixes two unrelated things and accesses both via magic negative
indices: per-qubit trainable rotation angles (`weights[start:start+3*N:3]`-style slicing) and a
handful of *auxiliary* scalars (scale factor, loss bias, Hamiltonian coefficients) stuffed into the
tail (`weights[-1]`, `weights[-2]`, `weights[-6:-1]`, `weights[-11:-1]` — inconsistent between
circuits, and `losses.py` independently hardcodes one of these slices). This refactor replaces both
with two explicit, named pieces:

**Rotation tensor** — shape declared as a tuple `(L, N, R)`:
- `L` — number of layers
- `N` — qubits addressed *independently* per layer (a circuit whose per-layer rotation is shared
  across all active wires, like QCNN, declares `N = 1` — one broadcast parameter set, not one bug)
- `R` — trainable rotation ops per qubit per layer

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

`rotate()` indexes into the tensor directly (`weights.rot[layer, :, op]`) instead of computing flat
offsets by hand; an op index `>= R` is a plain out-of-bounds error from that array access — no
separate check needed. `validate()` is called once per layer inside `build()` (see section 4), so a
config that requests more layers or wires than the circuit was sized for fails immediately with a
named reason instead of a shape-mismatch deep inside PennyLane.

**Named aux weights** — declared per circuit as `aux_defaults: dict[str, float]`, each entry's
*initial value* overridable from Hydra config:

```yaml
# hydra_configs/VQC/base.yaml
aux_weights:
  scale_factor: 1.0
  bias: 0.1
```

`train.py` merges `{**circuit.aux_defaults, **cfg.get('aux_weights', {})}` — no more flat-vector
slicing, no more the two circuits disagreeing on offsets (`-1` vs `-2`) purely because their
neighboring slices claimed different amounts of trailing space; both now just say
`weights.aux['scale_factor']`.

`hamiltonian_coeffs` (VQC's Hamiltonian, today's fixed `weights[-11:-1]`, 10 coefficients
regardless of qubit count) is the one aux entry whose correct length depends on a runtime value
(qubit count), so it isn't given a static class-level default. Instead `VQCCircuit.measure()`
checks it explicitly: use `cfg.aux_weights.hamiltonian_coeffs` if present (validating
`len(...) == len(wires)`, raising otherwise), else default to `[0.1] * len(wires)`. This is the
same historical fixed-10-vs-N-qubits question flagged during the earlier audit discussion, now
resolved as a named, length-checked config entry instead of a silent shape mismatch — confirmed via
the run archive that every past run using this Hamiltonian measurement used exactly 10 qubits, so
this changes no existing run's behavior, only what happens at other qubit counts (a clear error
instead of a confusing one).

Also noticed while migrating (unrelated to weights): `QuantumClassifier.__init__` /
`_initialize_wires` sets `use_ancilla`, `truth_wire`, `separate_ancilla`, none of which are read by
either circuit — likely vestigial from an earlier ancilla-based measurement scheme. Decide during
migration whether to carry them into the new protocol or drop them; don't carry them forward
silently unused a second time.

**Open risk, needs checking before implementation, not just design:** `QuantumTrainer` currently
calls `qml.AdamOptimizer.step_and_cost(cost_fn, weights)` on one flat trainable array. Splitting
weights into `rot` (an array) plus `aux` (named scalars) means the optimizer call must track
several trainable objects, e.g. `opt.step_and_cost(cost_fn, rot, *aux.values())`, then repack into
the `CircuitWeights` used by the next iteration — and checkpoint save/load (currently a flat-array
pickle) must serialize the same structure. This touches `QuantumTrainer`, not just
`quantum/circuits/`; confirm PennyLane's optimizer supports multiple trainable args cleanly before
committing to this shape in `architectures.py`.

## 4. The `Circuit` protocol

```python
# quantum/circuits/base.py
from dataclasses import dataclass
from typing import Protocol, List, Dict, Any
import pennylane as qml
import pennylane.numpy as np

@dataclass
class CircuitWeights:
    rot: np.ndarray          # shape (L, N, R) -- see section 3
    aux: Dict[str, Any]      # name -> value, from circuit.aux_defaults merged with cfg.aux_weights

class Circuit(Protocol):
    num_layers: int
    aux_defaults: Dict[str, float]

    def rotation_shape(self, n_qubits: int, n_layers: int) -> "RotationShape": ...

    def encode(self, weights: CircuitWeights, inputs: np.ndarray, layer: int, wires: List[int]) -> None:
        """State preparation from data. Takes `weights` so a circuit can read
        a named aux value (e.g. scale_factor) without any weights-free vs
        weights-taking ambiguity."""
        ...

    def entangle(self, wires: List[int]) -> None: ...

    def rotate(self, weights: CircuitWeights, layer: int, wires: List[int]) -> None: ...

    def measure(self, weights: CircuitWeights, wires: List[int]):
        """Terminal measurement. Return an expval (scalar or Hamiltonian) or a
        list of expvals."""
        ...

    def build(self, weights: CircuitWeights, inputs: np.ndarray, wires: List[int]):
        """Full circuit body; QNode is built directly from this method.
        Default composition below; override for a different loop shape."""
        shape = self.rotation_shape(len(wires), self.num_layers)
        for L in range(self.num_layers):
            shape.validate(L, wires)
            self.encode(weights, inputs, layer=L, wires=wires)
            self.entangle(wires)
            self.rotate(weights, layer=L, wires=wires)
        return self.measure(weights, wires)
```

`CircuitBase` is a concrete class implementing just `build()` as above, so a new "standard" circuit
subclasses it and only has to write `rotation_shape`, `aux_defaults`, `encode`, `entangle`,
`rotate`, `measure`. QCNN subclasses `Circuit` directly (or `CircuitBase` and overrides `build`)
because its loop shape differs.

## 5. VQC migrated

```python
# quantum/circuits/vqc.py
import pennylane as qml
import pennylane.numpy as np
from .base import CircuitBase, RotationShape, CircuitWeights
from helpers.utils import getIndex

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class VQCCircuit(CircuitBase):
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

    def rotation_shape(self, n_qubits, n_layers):
        return RotationShape(L=n_layers, N=n_qubits, R=3)   # per wire per layer: RZ, RY, RX

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
        rot_z = weights.rot[layer, :, 0]
        rot_y = weights.rot[layer, :, 1]
        rot_x = weights.rot[layer, :, 2]
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

`weights.aux['bias']` (today's `sum(weights[-6:-1])` in `losses.py`) is read at the loss-function
call site, not inside the circuit — same as today, just a named lookup instead of a slice; updating
that call site is a required knock-on change (see section 8), not part of `quantum/circuits/`.

## 6. QCNN migrated (owns `build()` — the loop shape genuinely differs)

```python
# quantum/circuits/qcnn.py
import pennylane as qml
import pennylane.numpy as np
from .base import Circuit, RotationShape
from helpers.utils import getIndex

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class QCNNCircuit(Circuit):
    aux_defaults = {'scale_factor': 1.0}

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'),
            'pt':  getIndex('particle', 'pt'),
        }

    def rotation_shape(self, n_qubits, n_layers):
        return RotationShape(L=n_layers, N=1, R=2)   # (phi, theta) shared across active wires

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
        # "wires" here is the pooled/active set for this layer -- see build() below
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

## 7. State-circuit derivation for QFI (replacing the hand-duplicated `_vqc_state_circuit` twin)

`quantum_fisher()` / `run_fisher_computation()` need a version of the circuit that returns
`qml.state()` instead of `measure()`'s expval. Today this is a hand-copied second method per
circuit. Since `build()` always ends by calling `self.measure(...)`, the state version is derived
generically instead — give `Circuit.build` an optional `measure_override: Callable | None = None`
parameter, called instead of `self.measure(...)` at the end when set:

```python
# quantum/architectures.py (QuantumClassifier)
def _state_build(self, weights: CircuitWeights, inputs, wires):
    return self._impl.build(weights, inputs, wires, measure_override=lambda *_: qml.state())
```

One circuit implementation now produces both the expval and state versions; a new circuit gets QFI
support for free as long as its `build()` calls `self.measure(...)` exactly once, at the end.

Note from section 3: `set_circuit()` currently wraps only the VQC state circuit in
`qml.defer_measurements(...)`, not the QCNN one — worth checking whether that's load-bearing
(neither circuit appears to use mid-circuit measurement today) before deciding whether the generic
wrapper needs it too.

## 8. Registry and `QuantumClassifier` integration

```python
# quantum/circuits/registry.py
from .vqc import VQCCircuit
from .qcnn import QCNNCircuit

_REGISTRY = {
    'normal': VQCCircuit,
    'CNN': QCNNCircuit,
}

def get(circuit_type: str, num_layers: int):
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

`train.py`'s weight construction (currently `NUM_WEIGHTS = ... + cfg.extra_weights`, a flat
`np.random.uniform` array with `init_weights[-cfg.extra_weights:-1] = 0.1`) becomes:

```python
shape = VQC._impl.rotation_shape(len(VQC.auto_wires), cfg.num_layers)
rot = np.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R))
aux = {**VQC._impl.aux_defaults, **dict(cfg.get('aux_weights', {}))}
init_weights = CircuitWeights(rot=rot, aux=aux)
```

`cfg.extra_weights` and its associated `train.py:109-121` branch are dropped entirely — every aux
weight is now declared by name (in the circuit's `aux_defaults` and/or Hydra config), so there's no
count to reconcile.

`losses.py`'s `VQC_cost`/`probabilistic_loss` call sites read `weights.aux['bias']` instead of
`sum(weights[-6:-1])` — a required signature change in `losses.py`, out of `quantum/circuits/`'s
own scope but a direct consequence of removing the flat vector.

## 9. Migration plan (incremental, non-breaking)

1. Add `quantum/circuits/base.py` (`RotationShape`, `CircuitWeights`, `Circuit` protocol,
   `CircuitBase` default `build()`). No existing code changes yet.
2. Add `quantum/circuits/vqc.py` and `quantum/circuits/qcnn.py` per sections 5-6.
3. Add `quantum/circuits/registry.py`.
4. Add an `aux_weights:` block to `hydra_configs/{VQC,QAE,VPC,QFI,AOJ}/base.yaml` (one key per
   circuit's `aux_defaults`).
5. Confirm the optimizer/checkpoint risk flagged in section 3 (`QuantumTrainer`'s
   `step_and_cost`/checkpoint code needs to handle `rot` + named `aux` as separate trainables, not
   one flat array) before relying on this shape.
6. Write a parity test: for a fixed seed, a `CircuitWeights` built to match today's flat weight
   values exactly, and a fixed input batch, assert the new `VQCCircuit().build(...)` and
   `QCNNCircuit().build(...)` produce identical output to the current `_vqc_circuit` /
   `_qcnn_implementation` (float tolerance). This is the actual safety net for the refactor.
7. Only once step 6 passes: switch `QuantumClassifier.set_circuit()` to the registry-based version
   (section 8), delete `_vqc_circuit`, `_vqc_state_circuit`, `_qcnn_implementation`,
   `_qcnn_state_circuit` from `architectures.py`, switch `train.py`'s weight construction, and
   update `losses.py`'s bias read.
8. Re-run one short training job (a handful of epochs, small config) end to end and confirm the
   loss curve and checkpoint format are unchanged (adjusted for the new `CircuitWeights`
   serialization) before relying on this for real runs.

Steps 1-4 touch no existing behavior and can be merged on their own. Step 7 is the only point where
the old code paths are removed — gate it on the parity test, not on inspection.
