# Circuit Module Refactor — Completed

Implemented deliverables from `REFACTOR.md` (companion: `AUDIT.md` section 3.2). Each entry below
is merged and verified; remaining work is in `REFACTOR.md`.

## D1 — Weight types and `Circuit` protocol

**Commit:** `47df1e6`. **File:** `quantum/circuits/base.py`.

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

**Verified:** `RotationShape.validate` raises on both layer overrun and wire overrun (checked
directly against fixed inputs).

## D2 — VQC circuit

**Commit:** `11df1bb`. **File:** `quantum/circuits/vqc.py`.

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

**Ancilla-attributes question (design doc's A4), resolved:** `use_ancilla`, `truth_wire`,
`separate_ancilla` are dropped, not carried into `Circuit`/`VQCCircuit` — nothing in `_vqc_circuit`
ever read them.

**Verified:** `VQCCircuit().build(...)` runs through a real `qml.QNode` and returns a scalar
expval; the `measure_override` hook (needed later by D6) already works against `build()`
unmodified.

## D4 — Registry

**Commit:** `622ddb4`. **File:** `quantum/circuits/registry.py`.

```python
from .vqc import VQCCircuit

_REGISTRY = {'normal': VQCCircuit}

def get(circuit_type: str, num_layers: int) -> Circuit:
    try:
        return _REGISTRY[circuit_type](num_layers=num_layers)
    except KeyError:
        raise ValueError(f"Unknown circuit_type '{circuit_type}'. Registered: {list(_REGISTRY)}")
```

Only `'normal'` is registered — `circuit_type='CNN'` raises `ValueError` rather than dispatching
anywhere; expected until QCNN is reintroduced as its own deliverable.

**Verified:** `registry.get('normal', 1)` returns a `VQCCircuit` instance; `registry.get('CNN', 1)`
raises `ValueError: Unknown circuit_type 'CNN'. Registered: ['normal']`.

## D5 — Hydra config fields

**Commit:** `31af96b`. **Files:** `hydra_configs/{VQC,QAE,VPC,QFI,AOJ}/base.yaml`.

Added, next to the existing `wires`/`circuit_type`:

```yaml
operations_per_qubit: 3        # VQC's RZ/RY/RX per wire per layer
aux_weights:
  scale_factor: 1.0
  bias: 0.1
```

The loose top-level `hydra_configs/base.yaml` was left untouched — it's byte-identical to
`VQC/base.yaml` but isn't referenced by any `config_path` in `train.py`/`evaluate.py`, so it looks
orphaned rather than load-bearing.

**Verified:** all five files carry both fields with correct YAML indentation (checked by eye —
no `yaml`/Hydra installed in this environment to parse them programmatically); a full
`registry.get()` → `rotation_shape()` → `CircuitWeights` → `qml.QNode` run confirms the values these
fields are meant to supply work end to end.

**Unrelated finding surfaced while verifying, not fixed:** `train.py:20`'s
`@hydra.main(config_path="./hydra_configs/VQC", config_name="config")` references a `config.yaml`
that has never existed in this repo (only `base.yaml`) — `python train.py` currently fails at
startup. Recorded in `AUDIT.md` section 2; not part of this refactor.
