# JAX backend feature plan

Status: planning only, nothing implemented. Branch `feature-jax`. Written 2026-09-10.

## 1. Scope

Add `backend: 'jax'` as a second, fully opt-in value for the existing `backend` config field
(`configs/base.yaml:23`), which is passed straight through as the PennyLane QNode `interface`
kwarg (`quantum/architectures.py:206`). `backend: 'autograd'` remains the default and its code
path, weight representation, optimizer, and checkpoint format must not change.

**In scope:**
- Analytic execution only (`shots <= 0`), `default.qubit`, `diff_method='backprop'` (the base
  config default) as the primary target, `diff_method='parameter-shift'` also supported since it
  works with `interface='jax'` on any device/shot count.
- `jax.jit`-compiled training step.
- A JAX-native optimizer (`optax.adam`) with its own checkpoint schema, fully separate from
  `qml.AdamOptimizer`'s.
- A new conda env spec (not created by this plan) with GPU (CUDA) JAX.

**Explicitly out of scope (documented for later, not built now):**
- `backend='jax'` with `shots > 0` (finite-shot sampling). Must be rejected with a clear
  `ValueError` at construction time, the same way incompatible `diff_method` combinations are
  already rejected (`quantum/architectures.py:202-218`).
- `adjoint` diff_method for `backend='jax'` for training (it silently zero-gradients
  `hamiltonian_coeffs` under `jax` exactly as it already does under `autograd` — verified below,
  not a new limitation, but still excluded from the maintained jax training path for the same
  reason `adjoint` is already discouraged for training in `llm_summary.MD`).
- Any change to `run_experiments.py`'s CLI or the autograd path's checkpoint/config schema.

## 2. Verified findings

All claims below were checked directly in this session, not inferred from documentation age.

**Machine environment (probed 2026-09-10):**
- `nvidia-smi`: driver 570.211.01, CUDA 12.8, 2x NVIDIA L40S (46 GB each).
- `nvcc --version`: CUDA compiler 12.8.93.
- Current env `pennylane-gpu-sep2026`: Python 3.14.7, PennyLane 0.45.1, PennyLane-Lightning
  0.45.0 (+ `_gpu`, `_kokkos` variants installed), NumPy 2.5.2, SciPy 1.18.1, autograd 1.8.0,
  OmegaConf 2.3.1, h5py 3.16.0, loguru 0.7.3, scikit-learn 1.9.0, tqdm 4.70.0, wandb 0.29.0,
  torch 2.14.0 (unused by the maintained pipeline but present).

**JAX/PennyLane version compatibility — critical, load-bearing finding:**
PennyLane 0.45.1's JAX interface code calls `jax.core.concrete_or_error` / `jax.core.is_concrete`.
Both were removed in JAX 0.11.0 (confirmed against the JAX changelog). Verified empirically in an
isolated throwaway venv (`pennylane==0.45.1` + `jax==0.11.1`, CPU):
- `diff_method='adjoint'` and `'parameter-shift'` under `interface='jax'` both raise
  `ImportError: cannot import name 'concrete_or_error' from 'jax.core'` at grad time.
- `jax.jit(jax.grad(cost_fn))` with `diff_method='backprop'` raises
  `AttributeError: jax.core.is_concrete was deprecated ... and removed in JAX v0.11.0`.
- Plain (non-jit) `diff_method='backprop'` gradients still work under `jax==0.11.1`.

Re-ran the same checks with `jax==0.10.2` (latest release before the breaking removal;
`jaxlib==0.10.2` and `jax-cuda12-plugin==0.10.2` both exist on PyPI) — **everything passes**:
`backprop`, `adjoint`, and `parameter-shift` all construct and run; `jax.jit(jax.grad(...))` with
`backprop` works; `optax` 0.2.8 (`jax>=0.5.3` required, satisfied) installs and runs cleanly.

**Conclusion: pin `jax==0.10.2` (and matching `jaxlib`/`jax-cuda12-plugin`) for this PennyLane
version. Do not let a plain `pip install jax`/`jax[cuda12]` float to latest — it will silently
break the jax backend the moment PennyLane's `jax-jit` interface path or `adjoint`/
`parameter-shift` under `jax` is exercised, while plain non-jit `backprop` gradients would
misleadingly still work, masking the incompatibility until a different diff_method or jit is
tried.** Recheck this pin whenever PennyLane is upgraded — this is a PennyLane 0.45.1-specific
constraint, not a permanent JAX ceiling.

**Hamiltonian-coefficient differentiability under `interface='jax'` (parity check against the
autograd-path finding in `llm_summary.MD` / problems.MD P9), verified with `jax==0.10.2`,
`default.qubit`, `jax_enable_x64=True`:**

| diff_method | interface='jax', `hamiltonian_coeffs` gradient |
|---|---|
| `backprop` | nonzero, correct (verified against known input) |
| `parameter-shift` | nonzero, correct, numerically identical to backprop's |
| `adjoint` | exact zero — same silent failure mode already documented for `autograd` |

This is exactly the existing autograd-path pattern, now confirmed to extend to the jax interface.
No new caveat; the existing repo guidance ("don't use `adjoint` for training this circuit")
applies unchanged to `backend='jax'`.

**`jax.jit` + `jax.grad` + `backprop` + `default.qubit`:** verified end-to-end, including with a
dict-of-arrays weight pytree (mirroring `CircuitWeights.rot`/`aux` shape) and an `optax.adam`
train step. See the exact pattern in section 5.

**`qml.transforms.broadcast_expand` (used unconditionally at `architectures.py:221`) under
`interface='jax'`:** verified working correctly for the real calling convention used by this
codebase — weights (`rot`, `aux` values including `hamiltonian_coeffs`) unbatched, only `inputs`
carrying the batch dimension (matches `quantum/losses.py:63-66`, `VQC_cost`'s
`quantum_circuit(weights, inputs)` call). No changes needed to `broadcast_expand` usage or to
`CircuitBase`/`VQCCircuit` for the jax backend.

**Precision:** JAX defaults to float32. `jax.config.update("jax_enable_x64", True)` must be set
process-wide before any JAX array is created, or all downstream computation silently runs in
float32 and will not numerically match the existing float64 `pennylane.numpy`/autograd path
(breaking golden-value parity tests). This is a global, one-time, process-lifetime flag — it
cannot be toggled per-array after the fact.

**Multi-process GPU memory (relevant to `run_experiments.py`):** JAX's default
(`XLA_PYTHON_CLIENT_PREALLOCATE=true`) claims ~75% of a GPU's memory in the first process to
touch it. `run_experiments.py` launches up to `--num-cores` concurrent training subprocesses;
if more than one lands on jax with default settings, the second process's allocation attempt can
OOM even though the GPU has free memory. Documented, supported mitigations: set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (allocate on demand) or
`XLA_PYTHON_CLIENT_MEM_FRACTION=<0.9/N>` for N expected concurrent processes per GPU. Must be set
as an environment variable before the JAX runtime initializes in each subprocess (i.e. before
`import jax` in that process).

**optax state and checkpointing:** `optax.adam(...)` returns a `GradientTransformation` whose
`init(params)` produces a nested `NamedTuple` pytree (`ScaleByAdamState(count, mu, nu)` plus an
`EmptyState` for the bias-correction/no-op stages in the default `adam` chain) holding JAX arrays
matching the `params` pytree structure. Verified: this pytree, and the params pytree, both pickle
correctly once converted to plain NumPy via `jax.tree_util.tree_map(np.asarray, tree)` before
`pickle.dump`, and rehydrate correctly via `jax.tree_util.tree_map(jnp.asarray, tree)` after
`pickle.load` — confirmed a restored optimizer continues training correctly (loss keeps
decreasing, momentum carries over) rather than resetting. Do not pickle live JAX `Array` objects
directly inside a payload dict without this numpy round-trip — JAX array pickling is
device/backend-dependent and not the documented portable path.

## 3. New conda environment spec

Do not create this env as part of the plan — for the implementor to create. Suggested name:
`pennylane-gpu-jax-sep2026` (kept separate from `pennylane-gpu-sep2026` so the maintained autograd
path's env is never put at risk by a jax-driven dependency conflict).

```bash
conda create -n pennylane-gpu-jax-sep2026 python=3.14
conda activate pennylane-gpu-jax-sep2026

# Same core stack as pennylane-gpu-sep2026 (pin exact versions to match)
pip install "pennylane==0.45.1" "pennylane-lightning==0.45.0" \
            "pennylane-lightning-gpu==0.45.0" "pennylane-lightning-kokkos==0.45.0" \
            "numpy==2.5.2" "scipy==1.18.1" "autograd==1.8.0" \
            "omegaconf==2.3.1" "h5py==3.16.0" "loguru==0.7.3" \
            "scikit-learn==1.9.0" "tqdm==4.70.0" "wandb==0.29.0"

# JAX, pinned -- see section 2 for why this exact version, not latest
pip install "jax[cuda12]==0.10.2" "optax"
```

Notes for the implementor:
- `jax[cuda12]==0.10.2` pulls `jaxlib==0.10.2`, `jax-cuda12-plugin==0.10.2`, and bundled
  `nvidia-*-cu12` wheels (dry-run resolved cleanly against CUDA 12.9 runtime wheels in this
  session). The installed driver (570.211.01, CUDA 12.8 capable) is expected to run CUDA 12.9
  runtime wheels under NVIDIA's minor-version forward compatibility, but this was not verified on
  actual GPU hardware in this session (the verification venv was CPU-only, no GPU present in the
  planning sandbox) — **the implementor must confirm a real GPU op runs** (e.g.
  `jax.numpy.ones(3).device`, then a small matmul) immediately after creating this env, before
  relying on it.
  If the driver rejects the bundled runtime, fall back to `pip install "jax[cuda12_local]==0.10.2"`,
  which uses the system CUDA toolkit already present (`nvcc` confirms 12.8.93 is installed).
- `optax` has no strict upper pin needed against `jax==0.10.2` (`optax`'s own floor is
  `jax>=0.5.3`); just avoid installing a newer `jax` afterward that would silently override the
  pin.
- Do not install `torch` in this env unless something explicitly needs it; the maintained
  pipeline doesn't use it.
- Confirm `python -m unittest discover -s tests -q` still collects and skips (not errors) inside
  this new env before using it for anything beyond jax-specific development, since some test
  infra imports may assume the primary env's package set.

## 4. File-by-file change list

**`configs/base.yaml`**
- No default change. Document that `backend: 'jax'` is now a valid opt-in value (comment only,
  mirroring the existing `diff_method` comment style).

**`configs/README.md`**
- Update the `backend` row: `'autograd'` (default, maintained) and `'jax'` (opt-in; requires
  `shots<=0`; requires the `pennylane-gpu-jax-sep2026` env; uses `optax.adam` instead of
  `qml.AdamOptimizer`, with a separate checkpoint schema — link to this plan file or fold a
  condensed version into `llm_summary.MD` once implemented).

**`quantum/architectures.py`**
- `QuantumClassifier.set_circuit()` (~line 202-221): add validation raising `ValueError` when
  `self.backend == 'jax' and self.device.shots is not None` (or however finite shots are exposed
  on the device/config — check `cfg.shots > 0` at construction, mirroring the existing
  diff_method `try/except` block's style and message quality). No other change needed here —
  `interface=self.backend` already passes `'jax'` straight through, and `broadcast_expand` is
  confirmed to work as-is (section 2).
- Add a small helper (e.g. `_enable_jax_x64_once()` or inline in `QuantumClassifier.__init__`)
  that calls `jax.config.update("jax_enable_x64", True)` exactly once, guarded (JAX warns/no-ops
  on repeat calls, but keep it explicit and idempotent) — only when `backend_name == 'jax'`, so
  importing `jax` at all stays conditional on this backend being selected (never import jax
  eagerly at module load for the autograd path).
- `CircuitWeights` construction / weight init (init happens in `train.py`, see below, but the
  container is defined here): keep the existing `pennylane.numpy`-based RNG init logic completely
  unchanged (it's what makes `random_seed` reproducible — don't fork it). Add a boundary
  conversion, e.g. `CircuitWeights.to_jax()` / a free function `to_jax_pytree(weights: CircuitWeights) -> dict`
  and its inverse, used only at the point weights are handed to the jax train step / read back
  from it. This keeps initialization identical and reproducible across both backends and
  localizes all jax-specific array handling to one narrow seam.
- `QuantumTrainer.iteration()` (~line 417-457): branch on `self.backend`. Keep the existing
  autograd branch (`self.optim.step_and_cost(...)`) untouched. Add a jax branch:
  ```python
  # illustrative pattern verified in this session, not final code
  def cost_fn(params, data, labels):
      weights = CircuitWeights(rot=params["rot"], aux={k: v for k, v in params.items() if k != "rot"})
      return self.quantum_loss(weights, inputs=data, labels=labels,
                                quantum_circuit=self.circuit, loss_type=self.loss_type)

  loss, grads = jax.value_and_grad(cost_fn)(self.jax_params, data, labels)
  updates, self.jax_opt_state = self.optim.update(grads, self.jax_opt_state, self.jax_params)
  self.jax_params = optax.apply_updates(self.jax_params, updates)
  ```
  Wrap the whole step in `jax.jit` (see section 5) once correctness is confirmed unjitted first.
- `QuantumTrainer.__init__`: accept an already-constructed `optax` `GradientTransformation` the
  same way it currently accepts a `qml.AdamOptimizer` instance via the `optimizer` kwarg — keep
  `self.optim` generic; add `self.jax_opt_state` / `self.jax_params`, initialized only when
  `self.backend == 'jax'`.
- `QuantumTrainer.save_checkpoint()` (~line 671-700) / `restore_checkpoint()` (~line 702-714):
  branch the `'optimizer'` payload block on backend. For `jax`:
  ```python
  'optimizer': {
      'name': 'optax_adam',
      'hyperparams': {'lr': ..., 'b1': ..., 'b2': ..., 'eps': ...},
      'opt_state': jax.tree_util.tree_map(np.asarray, self.jax_opt_state),
  }
  ```
  and store `self.jax_params` (numpy-converted) alongside/instead of `self.current_weights` for
  this backend, or keep `CircuitWeights` as the single source of truth and convert at load —
  prefer the latter (keeps `'weights'` schema backend-agnostic, only `'optimizer'` diverges).
  `restore_checkpoint()` needs a backend branch that rehydrates `self.jax_opt_state` via
  `jax.tree_util.tree_map(jnp.asarray, ...)` instead of setting `self.optim.stepsize`/etc.
- `helpers/trained_run.py`-facing validation of checkpoint provenance (resume's "Adam type,
  hyperparameters, moments, and step counter" check, per `llm_summary.MD`) needs a backend-aware
  branch too — validate optax state pytree structure/dtype/finiteness instead of the
  `qml.AdamOptimizer` attribute set. Locate the exact validation function (search
  `helpers/trained_run.py` for the resume checks referenced in `llm_summary.MD`'s Resumption
  section) before implementing — not read in this planning pass.

**`train.py`**
- Lines 207-217: branch optimizer construction on `cfg.backend`. For `'jax'`, construct
  `optax.adam(learning_rate=cfg.lr)` (resume case reads hyperparams from the saved checkpoint's
  new `optimizer.hyperparams` block instead of `optimizer_state['stepsize']` etc.).
- Ensure `jax.config.update("jax_enable_x64", True)` fires before any weight/data ever reaches a
  jax array — the cleanest place is inside `QuantumClassifier.__init__`/`set_circuit()` (see
  above) since that's the earliest point `backend` is known and used, well before the optimizer
  or training loop.
- If `run_experiments.py` subprocess launch is ever extended to set per-child env vars (see GPU
  memory pitfall in section 2/6), that plumbing lives in `run_experiments.py`, not `train.py`;
  confirm current subprocess launch mechanism there before implementing.

**`run_experiments.py`**
- No functional change required for correctness, but document (docstring/comment) the GPU memory
  pitfall from section 2, and, if `--num-cores > 1` is used with `backend: 'jax'`, set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or a computed `XLA_PYTHON_CLIENT_MEM_FRACTION`) in each
  child subprocess's environment before it starts. Locate the actual subprocess-spawn call before
  implementing (not read in this planning pass) — likely `subprocess.Popen`/`multiprocessing`
  with an `env=` kwarg is the natural place.

**`publish_reports.py`**
- Already reads and surfaces `cfg.get("backend")` (line 179) — no change expected, but add a
  smoke check (manual or as part of the new tests) that a `jax`-backend experiment's `config.yaml`
  publishes without error, since the reader only touches `config.yaml`/results, not raw
  checkpoints, and should be backend-agnostic already.

**`helpers/config.py`**
- Optional, low-priority: validate `backend` is one of `{'autograd', 'jax'}` at config-load time
  with a clear error for typos, mirroring how other fields are validated. Not required for
  correctness (an unsupported string already fails inside `qml.QNode(interface=...)` with a
  PennyLane error), but a repo-native `ValueError` is more consistent with the existing
  diff_method validation UX. Judgment call — implementor may skip if it adds more code than value.

**`docker/Dockerfile` / `docker/README.md`**
- Not touched. The jax env is a separate conda env for local GPU development per the user's
  explicit instructions in this session; Docker packaging for it is out of scope unless
  requested later.

## 5. `jax.jit` integration plan

Verified working pattern (session smoke test, CPU, `jax==0.10.2` + `optax==0.2.8`):

```python
@jax.jit
def train_step(params, opt_state, data, labels):
    loss, grads = jax.value_and_grad(cost_fn)(params, data, labels)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```

`params` should be a flat dict pytree (`{'rot': ..., 'scale_factor': ..., 'bias': ...,
'hamiltonian_coeffs': ...}`) — confirmed optax/jax handle an arbitrary dict-of-arrays pytree with
no special-casing needed. `cost_fn` closes over `self.circuit`, `self.loss_type` (both static
Python objects, fine to close over — they are not traced).

**What must stay outside the jitted region:**
- `tqdm` progress bars / any per-step `print`/logging — these need concrete Python values, not
  tracers; call `train_step` first, then log the returned (already-concrete after `.block_until_ready()`
  or simple use) `loss`.
- W&B logging (`wandb.log(...)`) — same reason; happens after `train_step` returns.
- The learning-rate decay / early-stopping logic (`min_epochs`, `decay_patience`, AUC-improvement
  check) — this inspects concrete validation-AUC history across epochs and conditionally mutates
  `self.optim`'s effective learning rate. For `optax`, "changing the learning rate" mid-training
  means either (a) reconstructing `optax.adam(new_lr)` and copying the existing `opt_state` across
  (works: `ScaleByAdamState` doesn't encode the learning rate, only counts/moments — verified its
  structure in section 2), or (b) using `optax.inject_hyperparams(optax.adam)(learning_rate=...)`
  so `opt_state` itself carries a mutable, jittable learning-rate field. Prefer (b) — it avoids
  reconstructing the optimizer object and keeps the decay logic a simple `opt_state` field update
  outside the jit boundary, without needing to recompile `train_step`. This is a design decision
  for the implementor to make explicitly (document the choice in the code), not something this
  plan mandates further.
- Any Python-level branching on `loss_type` ('MSE' vs 'BCE') must be resolved to a fixed function
  *before* entering the jitted region (i.e. pick `LOSS_FNS[loss_type]` once at trainer-construction
  time, not inside `cost_fn` on every call) — `loss_type` is a static Python string, not a traced
  value, so this is a non-issue as long as it's captured by closure once, not passed as a jit
  argument.
- Checkpoint writing (`save_checkpoint`) happens after `train_step` returns and after converting
  the resulting params/opt_state to numpy (section 2) — never inside the jitted function.

**Recompilation/shape pitfall:** `jax.jit` retraces whenever an argument's shape changes.
`data`/`labels` have a fixed `batch_size` for every batch except a smaller final partial batch per
epoch (per `llm_summary.MD`, "a smaller final batch receives the correct weight"). Since dataset
size and `batch_size` are fixed for a whole run, there are exactly two distinct batch shapes across
the entire training run (regular, and final-partial), so `train_step` retraces at most twice per
run, not once per epoch — flag this in a comment where `train_step` is defined so a future reader
doesn't mistake occasional retracing for a bug. Do not attempt to pad the final batch to avoid
this; it is cheap (compilation happens once) and padding would need explicit loss-masking to avoid
skewing the correctly-sample-weighted loss average.

## 6. Risks and pitfalls (ranked by how likely they are to bite)

1. **JAX version drift breaking PennyLane's jax interface silently** (section 2). Pin
   `jax==0.10.2` in the new env; add a one-line runtime assertion or test that fails loudly if a
   future `pip install -U` moves it past 0.10.x, since the failure mode without jit/adjoint is a
   silent partial success (plain backprop still works) followed by a confusing crash later when
   jit or parameter-shift is exercised.
2. **GPU memory preallocation under `run_experiments.py` concurrency** (section 2). Must be
   handled per-subprocess via environment variables before `import jax` in the child process, or
   concurrent jax-backend runs will intermittently OOM on a shared GPU.
3. **float32-vs-float64 numeric drift** if `jax_enable_x64` isn't set early enough, or is set
   after some other library (unlikely, but JAX warns this must happen "as early as possible in
   your program"). Would silently produce plausible-looking but numerically divergent results
   versus the autograd path's float64 baseline — exactly the kind of thing the golden-value parity
   test in section 7 is there to catch.
4. **Checkpoint/resume format divergence between backends.** The `'optimizer'` block's schema is
   now backend-dependent (`qml.AdamOptimizer` fields vs. `optax` state). Resume must reject
   attempting to resume a `jax`-backend run with `autograd` code path or vice versa — this should
   already be caught by resume's existing exact-config-equality check (`backend` is a config
   field), but add an explicit test (section 7) rather than relying on that as an accidental side
   effect.
5. **Pickling raw JAX arrays/optax state without the numpy round-trip** (section 2) — silently
   works in-process but can produce environment-dependent or unreadable pickles across machines/
   JAX versions. Always convert via `jax.tree_util.tree_map(np.asarray, ...)` before `pickle.dump`.
6. **`jax.jit` recompilation surprises** (section 5) — minor (bounded to 2 shapes/run) but worth a
   code comment so it isn't mistaken for a performance bug later.
7. **New conda env's CUDA runtime wheels vs. installed driver** — the bundled `nvidia-*-cu12`
   wheels pulled by `jax[cuda12]==0.10.2` target CUDA 12.9; this session could only confirm the
   pip dependency resolution succeeds, not that it runs on this machine's driver (570.211.01,
   advertised CUDA 12.8) — no GPU was available in the planning sandbox. Low risk (NVIDIA's CUDA
   minor-version compatibility guarantee should cover this) but the implementor must verify with
   an actual GPU op immediately after env creation, with `jax[cuda12_local]` as the documented
   fallback.
8. **`optax.inject_hyperparams` design choice for LR decay** (section 5) is left as an explicit
   implementor decision, not fully specified here — flagged so it isn't silently skipped.

## 7. Testing strategy

New tests, mirroring existing rigor:

- **`tests/test_circuit_parity.py` addition:** a golden-value test analogous to the existing
  autograd golden-expectation-value test, but with `interface='jax'`, `diff_method='backprop'`,
  `default.qubit`, `jax_enable_x64` enabled — assert the jax-interface QNode output matches the
  existing autograd-interface golden value to a tight tolerance (e.g. `rtol=1e-6`) for identical
  weights/inputs. Also add a `diff_method`/`shots>0` rejection test for `backend='jax'` mirroring
  the existing diff_method validation tests.
- **Hamiltonian-coefficient gradient test** (mirrors the P9 verification, now for `jax`): assert
  nonzero, correct `hamiltonian_coeffs` gradients under `backprop` and `parameter-shift` with
  `interface='jax'`, and exact zero under `adjoint` — codifying the section 2 finding as a
  permanent regression test so a future PennyLane/JAX upgrade that changes this behavior is
  caught immediately.
- **`tests/test_circuit_weights_checkpoint.py` addition:** round-trip test for the new optax-based
  checkpoint schema — init optax state, take a few `train_step`s, save (numpy-converted) payload,
  reload, resume training, assert loss continues decreasing / moments match rather than resetting
  (mirrors the section-2 smoke test already run in this session).
- **`tests/test_config.py` addition:** `backend='jax'` accepted; `backend='jax'` + `shots>0`
  rejected with a clear `ValueError`.
- **`tests/test_saved_run.py` / resume tests addition:** attempting to resume a `jax`-backend
  checkpoint under an `autograd`-configured run (or vice versa) is rejected — verify this is
  actually caught (likely already is, via config equality) rather than assumed.
- **`tests/test_end_to_end.py` addition:** a short real (or small synthetic) training run with
  `backend='jax'` that completes, checkpoints, and resumes, mirroring the existing autograd
  end-to-end coverage — keep it small (few epochs, tiny `n_signal`/`n_background`) to stay fast,
  consistent with "don't waste tokens running unrelated tests" but this *is* the directly relevant
  test for a new feature.
- Run only the new/affected tests during implementation; run the full suite once before the final
  commit, per this repo's existing "Before committing changes" checklist in `llm_summary.MD`.

## 8. Ordered implementation task list

1. Create `pennylane-gpu-jax-sep2026` conda env per section 3; verify a real GPU jax op runs.
2. Add the `backend='jax'` + `shots>0` rejection in `QuantumClassifier.set_circuit()` (small,
   isolated, testable first).
3. Add `jax_enable_x64` activation, gated on `backend=='jax'`, in `QuantumClassifier.__init__`.
4. Write the golden-value parity test and the Hamiltonian-coefficient gradient test (section 7) —
   do this before touching the training loop, since they validate the QNode-construction layer in
   isolation.
5. Add the `CircuitWeights` <-> jax-pytree boundary conversion helper(s).
6. Implement the unjitted jax branch in `QuantumTrainer.iteration()` (value_and_grad + optax
   update), reusing the existing autograd branch's control flow shape as a template. Get this
   correct and tested before adding `jax.jit`.
7. Implement the optax checkpoint schema in `save_checkpoint`/`restore_checkpoint`; write the
   checkpoint round-trip test (section 7).
8. Wire `train.py`'s optimizer construction (lines 207-217) to branch on `cfg.backend`.
9. Wrap the jax training step in `jax.jit` per section 5; decide and implement the LR-decay
   mechanism (`inject_hyperparams` recommended); verify no recompilation-per-epoch regression via
   a quick manual timing check (not a committed test).
10. Address `helpers/trained_run.py`'s resume-provenance validation for the optax schema (locate
    the exact function first — not identified precisely in this planning pass).
11. Document the GPU-memory-preallocation mitigation in `run_experiments.py` (comment at minimum;
    implement the per-subprocess env var if `--num-cores`+jax concurrency is actually exercised).
12. Update `configs/README.md` and `llm_summary.MD` to describe the shipped feature (per this
    repo's AGENTS.md convention of updating `llm_summary.MD` after changes).
13. Run the new/affected tests, then the full suite once; review `git diff --check` and the full
    diff before committing, per `llm_summary.MD`'s "Before committing changes" checklist.
14. One concise, imperative commit per logical change (config validation, weight boundary,
    trainer jax branch, checkpoint schema, docs), per this repo's Git conventions.

## 9. Open judgment calls made in this plan

- **`optax.inject_hyperparams` vs. optimizer reconstruction for LR decay** (section 5) — not
  fully resolved; left as an implementor decision with a recommendation.
- **Where exactly the `CircuitWeights`<->jax-pytree conversion boundary lives** (a method on
  `CircuitWeights` vs. a free function in `quantum/architectures.py`) — left to the implementor's
  judgment at implementation time; either satisfies the "keep init logic backend-agnostic, convert
  only at the seam" requirement.
- **`helpers/trained_run.py`'s exact resume-validation function for optax state** — not located in
  this planning pass (out of the research budget); flagged as step 10 for the implementor to find
  and adapt rather than guessed at here.
- Everything else in section 4 that says "locate before implementing" (the `run_experiments.py`
  subprocess-spawn call, the precise resume-validation function) is deliberately left unresolved
  rather than guessed, consistent with this session's verify-the-mechanism practice.
