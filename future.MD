# Future work

## Quantum Fisher Information

The current classifier does not expose Quantum Fisher Information or metric-tensor computation.
The circuit protocol retains `measure_override`, which can build a state-returning version of a
registered circuit when this feature is restored.

The known blockers are:

1. Full metric-tensor evaluation exposes an auxiliary-wire error with
   `qml.metric_tensor(approx=None)`.
2. The block-diagonal and adjoint implementations then fail because a tuple reaches code that
   expects an object with a `shape` attribute. The source of that tuple has not been established.

Reimplementation should proceed in this order:

1. Give the full metric tensor an explicit auxiliary wire and isolate the tuple-shape failure in
   the block-diagonal and adjoint paths.
2. Build the state circuit through `Circuit.build(..., measure_override=...)` so each registered
   circuit does not need a duplicate state implementation.
3. Differentiate with respect to `CircuitWeights.rot` while treating auxiliary weights as
   constants. Verify the rotation-only submatrix against a fixed reference result.
4. Define and test the required parameter ordering before exposing QFI output to plotting code.

The feature is complete when QFI runs on the supported PennyLane versions, has a documented output
shape and ordering, and is covered by a numerical regression test.
