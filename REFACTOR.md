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
| **rotate** | separate loop over all wires, after entangle, using `weights[3*L*N : 3*L*N+3*N]` sliced into rz/ry/rx | fused with entangle: per surviving wire-pair, `RZ`/`RY` on both members then one `CNOT`, using `weights[2*L]`, `weights[2*L+1]` |
| **wires touched per layer** | all `N` wires, every layer | shrinking set `auto_wires[:-(1+L)]` — one fewer wire each layer (pooling) |
| **measure** | one scalar: `expval` of a weighted `Hamiltonian` over all wires | a list: `expval(PauliZ(i))` per wire |

QCNN's rotate and entangle are not sequential per-layer steps over a fixed wire set — they're
interleaved per shrinking wire-pair, and its encoding isn't per-layer at all. A single master loop
that hardcodes "for each layer: encode, then entangle, then rotate" cannot express QCNN's pooling
topology without one of the four operations becoming a dumping ground for logic that belongs to a
different step. So:

- The four operations (`encode`, `entangle`, `rotate`, `measure`) are the right *unit* of reuse.
- The **composition** of those four — the loop shape — is not; it must stay overridable per
  circuit, with a default implementation covering the common case (repeat encode → entangle →
  rotate every layer over a fixed wire set, measure once) so that circuits shaped like VQC don't
  have to write any loop at all.

## 3. The `Circuit` protocol

```python
# quantum/circuits/base.py
from typing import Protocol, List
import pennylane as qml
import pennylane.numpy as np

class Circuit(Protocol):
    """
    Contract every circuit implementation must satisfy. `build()` has a usable
    default (see CircuitBase) for the common "repeat every layer, measure once"
    shape; override it only when the loop structure itself differs, not the
    gates inside it.
    """

    num_layers: int

    def n_weights(self, n_qubits: int, n_layers: int) -> int:
        """Total trainable weights this circuit needs, EXCLUDING auxiliary
        weights (bias, scale-factor, loss-side coefficients -- see section 6).
        Must be the single source of truth: train.py calls this instead of
        computing a weight count inline."""
        ...

    def encode(self, inputs: np.ndarray, layer: int, wires: List[int]) -> None:
        """State preparation from data. Called however many times build()
        decides to call it -- once, or once per layer."""
        ...

    def entangle(self, wires: List[int]) -> None:
        """Fixed (non-parameterized) entangling gates over the given wires."""
        ...

    def rotate(self, weights: np.ndarray, layer: int, wires: List[int]) -> None:
        """Trainable single-qubit rotations for this layer's active wires."""
        ...

    def measure(self, weights: np.ndarray, wires: List[int]):
        """Terminal measurement. Return an expval (scalar or Hamiltonian) or a
        list of expvals -- whatever the loss function this circuit pairs with
        expects."""
        ...

    def build(self, weights: np.ndarray, inputs: np.ndarray, wires: List[int]):
        """Full circuit body; QNode is built directly from this method.
        Default composition below; override for a different loop shape."""
        for L in range(self.num_layers):
            self.encode(inputs, layer=L, wires=wires)
            self.entangle(wires)
            self.rotate(weights, layer=L, wires=wires)
        return self.measure(weights, wires)
```

`CircuitBase` is a concrete class implementing just `build()` as above, so a new "standard" circuit
subclasses it and only has to write `n_weights`, `encode`, `entangle`, `rotate`, `measure` — five
short methods, no loop. QCNN subclasses `Circuit` directly (or `CircuitBase` and overrides `build`)
because its loop shape differs.

## 4. VQC migrated (fits the default `build()` unchanged)

```python
# quantum/circuits/vqc.py
import pennylane as qml
import pennylane.numpy as np
from .base import CircuitBase
from helpers.utils import getIndex

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class VQCCircuit(CircuitBase):
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'),
            'pt':  getIndex('particle', 'pt'),
        }

    def n_weights(self, n_qubits: int, n_layers: int) -> int:
        return 3 * n_qubits * n_layers

    def encode(self, inputs, layer, wires):
        N = len(wires)
        sf = 2 * np.pi * sigmoid(self._weights[-1]) + 1   # see section 6: aux slot
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
        N = len(wires)
        start = 3 * layer * N
        rot_z = weights[start + 0 : start + 3 * N : 3]
        rot_y = weights[start + 1 : start + 3 * N : 3]
        rot_x = weights[start + 2 : start + 3 * N : 3]
        for rz, ry, rx, w in zip(rot_z, rot_y, rot_x, wires):
            qml.Rot(0., ry, rz, wires=w)
            qml.RX(rx, wires=w)

    def measure(self, weights, wires):
        obs = [qml.PauliZ(i) for i in wires]
        coeffs = weights[-11:-1]          # see section 6: aux slot, size mismatch flagged
        return qml.expval(qml.Hamiltonian(coeffs, obs))
```

`self._weights` in `encode` above is a small wrinkle: the current `_vqc_circuit` reads the
scale-factor out of `weights[-1]` inside the per-layer loop, meaning `encode` needs the full
weight vector even though its declared signature (per the protocol) doesn't take one. Two ways to
resolve this cleanly during migration, pick one:

- **(a)** widen `encode`'s signature to `encode(self, weights, inputs, layer, wires)` — the
  protocol takes `weights` in every method, some circuits just ignore it; or
- **(b)** keep `encode` weight-free and have `build()` compute `sf` once and pass it down (cleaner,
  and matches the fact that `sf` is a per-circuit-invocation constant, not something `encode`
  should recompute per wire). Recommended: **(b)**, requires overriding `build()` in `VQCCircuit`
  to precompute `sf` and thread it through — a two-line override, not a sign the loop shape itself
  differs.

## 5. QCNN migrated (owns `build()` — the loop shape genuinely differs)

```python
# quantum/circuits/qcnn.py
import pennylane as qml
import pennylane.numpy as np
from .base import Circuit
from helpers.utils import getIndex

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class QCNNCircuit(Circuit):
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'),
            'pt':  getIndex('particle', 'pt'),
        }

    def n_weights(self, n_qubits: int, n_layers: int) -> int:
        return 2 * n_layers

    def encode(self, inputs, layer, wires):
        sf = 2 * np.pi * sigmoid(self._weights[-2]) + 1   # note: -2, not -1 -- see section 6
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
        phi, theta = weights[2 * layer], weights[2 * layer + 1]
        for w in wires:
            qml.RZ(phi, wires=w)
            qml.RY(theta, wires=w)
            qml.RZ(phi, wires=(w + layer + 1) % N)
            qml.RY(theta, wires=(w + layer + 1) % N)
            qml.CNOT(wires=[w, (w + 1 + layer) % N])

    def measure(self, weights, wires):
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    def build(self, weights, inputs, wires):
        self._weights = weights   # see section 6
        self.encode(inputs, layer=0, wires=wires)   # once, not per layer
        self.entangle(wires)                          # initial ring only
        active = list(wires)
        for L in range(self.num_layers):
            active = active[:-(1 + L)]                 # pooling: one fewer wire per layer
            self.rotate(weights, layer=L, wires=active)
        return self.measure(weights, wires)            # measured over ALL original wires
```

## 6. Weight-layout convention (uncovered while migrating, needs a decision)

The current code encodes several *auxiliary* meanings into trailing slices of the flat weight
vector, and the convention is not consistent between the two circuits:

| Purpose | VQC | QCNN |
|---|---|---|
| Scale factor (`sf`) fed into `encode` | `weights[-1]` | `weights[-2]` |
| Bias added to score, in `losses.VQC_cost` | `sum(weights[-6:-1])` | same call site, same slice — applies regardless of circuit |
| Hamiltonian coefficients, in `measure` | `weights[-11:-1]` (10 fixed coefficients, independent of `n_qubits`) | n/a — QCNN measures per-wire, no Hamiltonian |
| `cfg.extra_weights` | appended after the circuit's own `n_weights()`, initialized to `0.1` except the last one | same mechanism, same `train.py` code path |

Two things worth resolving as part of this refactor, not after it:

1. **`weights[-11:-1]` is a fixed-size slice (10 coefficients) regardless of `n_qubits`.** Since
   `measure()` builds one `PauliZ` observable per wire in `wires`, this only lines up when
   `len(wires) == 10`. Worth confirming whether this is intentional (Hamiltonian only ever
   evaluated on a fixed 10-wire subset) or a latent bug that happens not to have been hit outside
   10-qubit configs. Flagging rather than "fixing" here since it changes model behavior.
2. **`sf` lives at a different offset per circuit** (`-1` vs `-2`) purely because `VQC_cost`'s bias
   slice (`-6:-1`) and `measure`'s Hamiltonian slice (`-11:-1`) claim different amounts of trailing
   space. Once circuits are split into their own files, give this a name instead of a magic
   negative index — e.g. each circuit declares `aux_weights: dict[str, slice]` (`{'scale_factor':
   slice(-1, None), ...}`) so `encode`/`measure` and `train.py`'s `NUM_WEIGHTS` computation read
   the same declared layout instead of independently-maintained magic numbers.

Also noticed in `QuantumClassifier.__init__` / `_initialize_wires`: `use_ancilla`, `truth_wire`,
and `separate_ancilla` are all initialized but never referenced inside either `_vqc_circuit` or
`_qcnn_implementation`. Likely vestigial from an earlier ancilla-based measurement scheme. Decide
during migration whether to carry them into the new `Circuit` protocol (if a future circuit needs
an ancilla) or drop them (if not) — don't carry them forward silently unused a second time.

## 7. State-circuit derivation for QFI (replacing the hand-duplicated `_vqc_state_circuit` twin)

`quantum_fisher()` / `run_fisher_computation()` need a version of the circuit that returns
`qml.state()` instead of `measure()`'s expval. Today this is a hand-copied second method per
circuit (`_vqc_state_circuit` duplicates `_vqc_circuit` gate-for-gate). Since `build()` always ends
by calling `self.measure(...)`, the state version can be derived generically instead:

```python
# quantum/architectures.py (QuantumClassifier), replacing the two hand-written *_state_circuit methods
def _state_build(self, weights, inputs, wires):
    with unittest.mock.patch.object(type(self._impl), 'measure', lambda self, w, wires: qml.state()):
        return self._impl.build(weights, inputs, wires)
```

(Or, more in the spirit of the codebase's existing style than `unittest.mock`: give `Circuit.build`
an optional `measure_override: Callable | None = None` parameter that, when set, is called instead
of `self.measure(...)` at the end — same effect, no monkeypatching.) Either way, **one** circuit
implementation now produces both the expval and state versions; a new circuit gets QFI support for
free as long as its `build()` calls `self.measure(...)` exactly once, at the end.

Note from section 6: `set_circuit()` currently wraps only the VQC state circuit in
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

`QuantumClassifier.set_circuit()` collapses from ~30 lines of `if circuit_type == 'CNN': ... else:
...` duplicating the QNode-construction call, to:

```python
def set_circuit(self, circuit_type: str = 'normal') -> None:
    self._impl = circuits.registry.get(circuit_type, self.num_layers)
    self.circuit = qml.QNode(self._impl.build, self.device, interface=self.backend)
    self.state_circuit_qnode = qml.QNode(self._state_build, self.device, interface=self.backend)
```

`train.py`'s `NUM_WEIGHTS` computation (currently lines 115-117, branching on `cfg.circuit_type`
itself) becomes:

```python
NUM_WEIGHTS = VQC._impl.n_weights(len(VQC.auto_wires), cfg.num_layers) + cfg.extra_weights
```

— one call, no duplicated `if circuit_type == 'CNN'` branch to keep in sync with the one inside
`architectures.py`.

## 9. Migration plan (incremental, non-breaking)

1. Add `quantum/circuits/base.py` (`Circuit` protocol + `CircuitBase` default `build()`). No
   existing code changes yet.
2. Add `quantum/circuits/vqc.py` and `quantum/circuits/qcnn.py` per sections 4-5, resolving the
   `encode`-needs-weights wrinkle (section 4) and naming the auxiliary weight slices (section 6)
   as part of writing them, not after.
3. Add `quantum/circuits/registry.py`.
4. Write a short parity test: for a fixed seed, fixed weights, and a fixed input batch, assert the
   new `VQCCircuit().build(...)` and `QCNNCircuit().build(...)` produce identical output to the
   current `QuantumClassifier._vqc_circuit` / `_qcnn_implementation` (bitwise or to float
   tolerance). This is the actual safety net for the refactor — everything above is a reshuffle of
   existing gate sequences, and this test is what proves nothing moved.
5. Only once step 4 passes: switch `QuantumClassifier.set_circuit()` to the registry-based version
   (section 8), and delete `_vqc_circuit`, `_vqc_state_circuit`, `_qcnn_implementation`,
   `_qcnn_state_circuit` from `architectures.py`.
6. Switch `train.py`'s `NUM_WEIGHTS` computation to `VQC._impl.n_weights(...)`.
7. Re-run one short training job (a handful of epochs, small config) end to end and confirm the
   loss curve and checkpoint format are unchanged before relying on this for real runs.

Steps 1-4 touch no existing behavior and can be merged on their own. Step 5 is the only point where
the old code paths are removed — gate it on the parity test, not on inspection.
