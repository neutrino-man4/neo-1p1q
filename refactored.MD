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

## D6 — QFI state-circuit derivation

**Commit:** `c5f43c6`. **File:** `quantum/architectures.py` (additive: one new method).

```python
def _state_build(
    self,
    weights: CircuitWeights,
    inputs: Optional[np.ndarray] = None,
    wires: Optional[List[int]] = None,
) -> Any:
    """State version of a registry circuit, derived generically for QFI."""
    wires = wires if wires is not None else self.auto_wires
    return self._impl.build(weights, inputs, wires, measure_override=lambda *_: qml.state())
```

Replaces the hand-duplicated `_vqc_state_circuit`/`_qcnn_state_circuit` pattern: any `Circuit`'s
`build()` already ends with exactly one `measure()` call, so swapping that call for `qml.state()`
via `measure_override` gives the state version for free. Not wired into `set_circuit()` yet — D9.

**Verified:** matches `_vqc_state_circuit`'s output exactly (see D8's `test_state_matches`).

**Later removed (2026-09-05):** `_state_build` and everything that wired to it existed solely to
serve `quantum_fisher()`/`run_fisher_computation()`, which turned out to be broken independent of
this refactor. Removed along with the rest of QFI — see `future.MD`. This entry is kept as the
historical record of what was built and verified at the time.

## D7 — Trainer support for split weights

**Commits:** `44b4551` (checkpoint round-trip test), `a89df22` (`load_weights` support). **Files:**
`tests/test_circuit_weights_checkpoint.py`, `quantum/architectures.py` (`QuantumClassifier`).

Two things needed confirming before D9 could safely restructure the trainer: whether PennyLane's
optimizer accepts multiple trainable arguments at once (rather than one flat array), and whether a
`CircuitWeights` instance survives a save/load round trip losslessly.

**Optimizer spike (throwaway, not committed):** built a small `rot` array plus two named scalars,
all `requires_grad=True`, and confirmed `qml.AdamOptimizer(0.1).step_and_cost(cost_fn, rot,
scale_factor, bias)` returns and updates all three arguments correctly, against the pinned
PennyLane 0.37.0 — `rot changed: True`, `scale_factor changed: True`, `bias changed: True`.

**Checkpoint round trip (`tests/test_circuit_weights_checkpoint.py`):** a `CircuitWeights` with a
trainable `rot` array and a trainable `aux` dict, pickled and unpickled via the same
`helpers.utils.Pickle`/`Unpickle` functions `QuantumTrainer.save()` already uses — `rot` and every
`aux` value compare equal after the round trip.

**`QuantumClassifier.load_weights()`** — additive branch only:
```python
saved_weights = dictionary['weights']
if isinstance(saved_weights, CircuitWeights):
    self.current_weights = CircuitWeights(
        rot=np.array(saved_weights.rot, requires_grad=train),
        aux={k: np.array(v, requires_grad=train) for k, v in saved_weights.aux.items()},
    )
    return
self.current_weights = np.array(saved_weights, requires_grad=train)
# ... original flat-array path unchanged below
```

**Verified:** old-format flat-array checkpoints still load bit-identically (`requires_grad`
preserved, same tensor type, `rearrange` path untouched); new-format `CircuitWeights` checkpoints
load correctly (`rot` values match, `requires_grad` correctly re-enabled on `rot` and every `aux`
value).

**Scope note carried into D9:** `QuantumTrainer.iteration()`'s live optimizer call was
deliberately left untouched here — it still passes one flat `current_weights` argument. Wiring the
confirmed multi-arg pattern into the real training loop needs `losses.py` to accept
`CircuitWeights` first, which is D9's job (see `REFACTOR.md`, D9 step 5).

## D8 — Parity test

**Commits:** `c5f43c6`'s test scaffolding is unrelated; test itself added alongside D6.
**File:** `tests/test_circuit_parity.py` (plus empty `tests/__init__.py` so `unittest discover`
can import the package).

Two `unittest` cases, both run through a real `qml.QNode` against the same device/backend:

- `test_expval_matches` — `VQCCircuit.build()` vs. today's `_vqc_circuit`, same converted weights,
  same input, compared to 10 decimal places.
- `test_state_matches` — `_state_build` (D6) vs. today's `_vqc_state_circuit`, full state vectors
  compared with `atol=1e-10`.

A helper, `_flat_to_circuit_weights`, converts one flat weight vector into the equivalent
`CircuitWeights` by mirroring `_vqc_circuit`'s own slicing (`rot` from the leading
`3*n_qubits*num_layers` entries reshaped to `(L, N, 3)`; `aux['scale_factor'] = flat[-1]`;
`aux['hamiltonian_coeffs'] = flat[-11:-1]`), so both sides run on identical weight values.

**Found and fixed during setup:** the test's `n_qubits` had to be fixed at exactly 10 — at any
other qubit count, the *old* `_vqc_circuit`'s Hamiltonian measurement itself raises
`ValueError: Could not create valid LinearCombination` (PennyLane), reproducing in a controlled
test the fixed-10-Hamiltonian-coefficient issue already flagged in `AUDIT.md`/`REFACTOR.md`. Not a
bug introduced by this refactor — the old code was never runnable at other qubit counts with this
measurement.

**Verified:** both tests pass. This is the actual safety net gating D9 (not just code inspection).

**`test_state_matches` later removed (2026-09-05):** its subject, `_state_build`, was removed along
with the rest of QFI — see the D6 note above and `future.MD`. `test_expval_matches` (the real
circuit-body/wiring guard, unrelated to QFI) remains and still passes.

## D9 — Switchover

**Commits:** `75f359e` (`architectures.py`), `81aa732` (`losses.py`), `08a2385` (`train.py`),
`00f73ed` (dead config field), `9a8870c` (parity test). **Files:** `quantum/architectures.py`,
`quantum/losses.py`, `train.py`, `hydra_configs/*/base.yaml`, `tests/test_circuit_parity.py`.

**`quantum/architectures.py`:**
- Deleted `_vqc_circuit`, `_vqc_state_circuit`, `_qcnn_implementation`, `_qcnn_state_circuit`, and
  the module-level `sigmoid()` helper they were the only callers of.
- `set_circuit()` now builds from the registry:
  ```python
  def set_circuit(self, circuit_type: str = 'normal', operations_per_qubit: Optional[int] = None) -> None:
      self._impl = registry.get(circuit_type, self.num_layers)
      if operations_per_qubit is not None:
          self._impl.operations_per_qubit = operations_per_qubit
      self.circuit = qml.QNode(
          lambda weights, inputs: self._impl.build(weights, inputs, self.auto_wires),
          self.device,
          interface=self.backend
      )
      self.state_circuit_qnode = qml.QNode(self._state_build, self.device, interface=self.backend)
  ```
  `circuit_type='CNN'` now raises `ValueError` from the registry (confirmed) — QCNN is fully gone,
  not just unmigrated. The `qml.defer_measurements` wrapper the old VQC state-circuit path used is
  dropped: D8's own passing test already proved `_state_build` (built without that wrapper) matches
  the wrapped legacy output bit-for-bit, so it was vestigial for this circuit.
- `QuantumTrainer.iteration()`'s training branch now uses D7's confirmed multi-arg optimizer
  pattern for real: `rot` and each named `aux` value are passed to `step_and_cost` as separate
  trainable arguments (via a small `cost_fn` closure that repacks them into a `CircuitWeights` for
  the actual loss function), and the returned tuple is repacked back into `self.current_weights`.
  The validation branch (`train=False`) was left untouched — it already worked with a single
  `CircuitWeights` argument once `losses.py` read `weights.aux['bias']`.

**`quantum/losses.py`:** `VQC_cost` and `batched_VQC_cost` (`probabilistic_loss` never read a bias)
now read `weights.aux['bias']` instead of `sum(weights[-6:-1])`.

**`train.py`:** `VQC.set_circuit(...)` now passes `operations_per_qubit=cfg.operations_per_qubit`.
Weight construction replaced with:
```python
shape = VQC._impl.rotation_shape(len(VQC.auto_wires), cfg.num_layers)
rot = np.array(np.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R)), requires_grad=True)
aux = {**VQC._impl.aux_defaults, **dict(cfg.get('aux_weights', {}))}
aux = {k: np.array(v, requires_grad=True) for k, v in aux.items()}
init_weights = CircuitWeights(rot=rot, aux=aux)
```
The `cfg.extra_weights > 8` sanity check and `NUM_WEIGHTS`/`init_weights[-cfg.extra_weights:-1]`
block are gone. The now-unconsumed `extra_weights: null` field was also removed from all six
`hydra_configs/*/base.yaml` files.

**`tests/test_circuit_parity.py`:** repurposed (see the test file's own docstring) — it compared
against the now-deleted `_vqc_circuit`/`_vqc_state_circuit` methods, so it now checks
`set_circuit()`'s wiring against a standalone `VQCCircuit` instead. The original old-vs-new
comparison's passing result is preserved above under D8.

**Verified:**
- All three tests pass (`test_circuit_parity.py` x2, `test_circuit_weights_checkpoint.py`).
- `quantum.architectures`, `quantum.losses`, `train.py`, `evaluate.py` all compile.
- A full `set_circuit` → `CircuitWeights` construction → one training `iteration()` → one
  validation `iteration()` round trip runs end to end (batch_size=1, matching `evaluate.py`'s own
  documented constraint) with real values, confirming `aux` values actually update after a step.
- `circuit_type='CNN'` now raises `ValueError: Unknown circuit_type 'CNN'. Registered: ['normal']`.
- `train.py` still fails at exactly the pre-existing, already-flagged `config_name="config"` bug —
  nothing new broken.
- `evaluate.py --cfg job` resolves cleanly: `operations_per_qubit`, `aux_weights`, `circuit_type`
  present and correct; no `extra_weights` key remains anywhere in the resolved config.

**Found while verifying, not fixed (out of D9's scope) — see `REFACTOR.md` D12:**
`quantum_fisher`/`run_fisher_computation` still assume a flat weight vector and pass
`self.current_weights` to PennyLane's `metric_tensor` as a single differentiable argument;
confirmed by direct test to raise `AttributeError` against a real `CircuitWeights`. Also noted
there: `QuantumTrainer.iteration()`'s validation branch only works at `batch_size=1` — pre-existing,
untouched by this change.

## D11 — Loss function registry

**Commit:** `6cdc614`. **File:** `quantum/losses.py`.

`VQC_cost` and `batched_VQC_cost` each hardcoded an identical `if loss_type=='BCE': ... elif
loss_type=='MSE': ... else: sys.exit(-1)`. Replaced with one module-level lookup:

```python
LOSS_FNS: Dict[str, Callable] = {
    'MSE': mfunc.mean_squared_error,
    'BCE': mfunc.binary_cross_entropy,
}
```

Both functions now do `score = exp_vals + bias` unconditionally, then
`LOSS_FNS[loss_type](labels, score)` (raising `ValueError` listing valid names on an unknown
`loss_type`, replacing the old unrecoverable `sys.exit(-1)`). Adding a new loss is now "write the
function, add one dict entry" — no existing cost function is touched.

**Correction made to the original D11 write-up:** it proposed folding `probabilistic_loss` into the
same lookup too. Re-checked against the actual code: `probabilistic_loss` takes a `loss_type`
parameter but never branches on it, and its scoring (indexing a probability vector by label) isn't
the same "labels, score -> scalar" shape as MSE/BCE. Left untouched; the dead `loss_type` parameter
is noted, not fixed, matching how other found-but-out-of-scope issues have been handled all
session. Now-unused `sys` import removed from `losses.py`.

**Verified:** `quantum/losses.py` compiles; MSE and BCE costs/scores are numerically unchanged
(same values as the D9 verification run: `MSE: cost=0.792801 score=0.109606`,
`BCE: cost=2.210861`); an unknown `loss_type` now raises `ValueError: Unknown loss_type 'bogus'.
Registered: ['MSE', 'BCE']` from both `VQC_cost` and `batched_VQC_cost`; all 3 existing tests still
pass; `train.py`/`evaluate.py`/`quantum/architectures.py` all still compile; `train.py` still fails
at exactly the same pre-existing `config_name` bug.

## D12 (attempted) — QFI removed entirely instead of fixed

**Commits:** `f5151b8` (`architectures.py` + `hydra_configs/QFI/` + `plotters/plot_utils.py`),
`ba81d28` (test file). **Files:** `quantum/architectures.py`, `hydra_configs/QFI/base.yaml`
(deleted), `plotters/plot_utils.py` (deleted), `tests/test_circuit_parity.py`.

D12 was originally scoped as "make `quantum_fisher`/`run_fisher_computation` work with
`CircuitWeights`." Verifying that before implementing (this session's usual practice) overturned
the diagnosis: I reconstructed the literal pre-D9 `_vqc_state_circuit` body (flat weight vector,
no `CircuitWeights` involved) and ran it through `qml.metric_tensor`/`qml.adjoint_metric_tensor`
at `batch_size=1` — the only way `quantum_fisher()` is ever called. **It fails identically, with
the same `AttributeError: 'ArrayBox' object has no attribute 'item'`, on the old, untouched,
flat-vector code.** The real cause is `encode()`'s `if inputs.shape[0] == 1: ... .item()` branch
breaking under PennyLane's QFI tracing, which wraps classical preprocessing in autograd `ArrayBox`
even for arguments not being differentiated — nothing to do with the weight representation, and
predates this entire refactor. Removing the `.item()` calls clears that crash but immediately
surfaces at least two more distinct errors (a missing auxiliary wire for `metric_tensor(approx=None)`,
and an unexplained `AttributeError: 'tuple' object has no attribute 'shape'` for `approx='block-diag'`
and `adjoint_metric_tensor`), left un-root-caused.

Given this, the user chose to remove QFI/metric-tensor computation entirely rather than
investigate or attempt a fix — see `future.MD`. Removed:
- `QuantumClassifier.quantum_fisher()`, `run_fisher_computation()`.
- `_state_build`, `state_circuit_qnode`, `state_circuit()` — these existed solely to serve QFI (no
  other caller anywhere in the codebase, confirmed by search).
- The `fisher_computation`/`approx` constructor params and the `aux_wire`/auxiliary-wire branch in
  `_initialize_wires()` (the ancilla-related `use_ancilla`/`truth_wire` branch, unrelated to QFI,
  was left alone).
- `hydra_configs/QFI/base.yaml` (the whole directory) and `plotters/plot_utils.py` (its only
  function plotted a QFI matrix; unimported anywhere, confirmed by search).
- `tests/test_circuit_parity.py`'s `test_state_matches` (its subject, `_state_build`, is gone;
  `test_expval_matches` is unrelated to QFI and stays).

**Kept, deliberately:** `Circuit.build()`'s generic `measure_override` parameter (in
`quantum/circuits/base.py`) — it doesn't mention QFI anywhere in its own code, is zero-cost, and
is exactly the hook a future metric-tensor re-implementation would still want; removing it would
make "implemented later" (see `future.MD`) harder, not easier.

**Verified:** `quantum/architectures.py` compiles and imports; a full `set_circuit` -> weight
construction -> train step -> validation step round trip still runs and produces the exact same
cost value as before (`0.7928011381077116`), confirming zero behavioral change to the parts that
remain; `no hasattr(VQC, 'state_circuit_qnode')` and `no hasattr(VQC, 'quantum_fisher')` both
confirmed True; a repo-wide search for `QFI`/`metric_tensor`/`quantum_fisher`/`fisher_computation`/
`adjoint_metric` turns up nothing left in any `.py`/`.yaml` file except `auto_qml.py` (a separate,
untouched legacy pipeline whose `metric_tensor` reference is PennyLane's `QNGOptimizer` machinery,
not the QFI-matrix feature being removed here) and the design docs themselves; `train.py` still
fails at exactly the same pre-existing `config_name` bug.
