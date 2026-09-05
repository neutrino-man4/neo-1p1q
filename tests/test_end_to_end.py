"""
D10: end-to-end validation. train.py itself can't run (its pre-existing,
deliberately-unfixed config_name bug, see AUDIT.md), so this exercises the
same pieces train.py/evaluate.py wire together -- QuantumTrainer's full
run_training_loop, a real checkpoint save/load round trip, and the
evaluate.py-equivalent inference pipeline -- built directly, without hydra.

Includes tests expected to fail: they pin down bugs that are pre-existing
and out of this refactor's scope (flagged, not fixed, see REFACTOR.md).
Author: Aritra Bal (ETP)
2026-09-05
"""
import os
import tempfile
import unittest

import numpy as onp
import pennylane as qml
import pennylane.numpy as np

import helpers.utils as ut
import quantum.architectures as arch
import quantum.losses as loss
from quantum.circuits.base import CircuitWeights


class _FixedBatches:
    """A minimal (data, labels) iterable standing in for a real DataLoader."""

    def __init__(self, data_batches, label_batches):
        self._data = data_batches
        self._labels = label_batches

    def __iter__(self):
        return iter(zip(self._data, self._labels))

    def __len__(self):
        return len(self._data)


def _make_batches(n_batches: int, batch_size: int, n_qubits: int, n_layers: int) -> _FixedBatches:
    """n_batches batches with alternating labels, so roc_auc_score always sees both classes."""
    data = [
        np.array(onp.random.uniform(-1, 1, size=(batch_size, n_qubits * n_layers, 3)))
        for _ in range(n_batches)
    ]
    labels = [onp.array([i % 2] * batch_size) for i in range(n_batches)]
    return _FixedBatches(data, labels)


def _build_trainer(save_dir: str, batch_size: int, epochs: int, n_qubits: int = 4, n_layers: int = 1):
    """Construct a QuantumClassifier + QuantumTrainer pair, ready to train."""
    onp.random.seed(0)
    vqc = arch.QuantumClassifier(
        wires=n_qubits, shots=None, dev_name='default.qubit',
        layers=n_layers, backend_name='autograd', test=False,
    )
    vqc.set_circuit('normal', operations_per_qubit=3)

    shape = vqc._impl.rotation_shape(n_qubits, n_layers)
    rot = np.array(onp.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R)), requires_grad=True)
    aux = {k: np.array(v, requires_grad=True) for k, v in vqc._impl.aux_defaults.items()}
    init_weights = CircuitWeights(rot=rot, aux=aux)

    optimizer = qml.AdamOptimizer(stepsize=0.05)
    trainer = arch.QuantumTrainer(
        model=vqc, lr=0.05, optimizer=optimizer, loss_fn=loss.VQC_cost,
        save=True, train_max_n=batch_size * 2, valid_max_n=batch_size * 2,
        epochs=epochs, wandb=None, loss_type='MSE',
        init_weights=init_weights, batch_size=batch_size, logger=__import__('loguru').logger,
    )
    trainer.set_directories(save_dir)
    return vqc, trainer


class TestFullTrainingLoop(unittest.TestCase):
    """D10: run_training_loop must run end to end and produce a usable checkpoint."""

    def test_runs_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=2)
            train_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)
            val_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)

            history = trainer.run_training_loop(train_loader, val_loader)

            self.assertEqual(len(history['train']), 2)  # one per epoch, epoch 0 is val-only
            self.assertEqual(len(history['val']), 3)     # epoch 0 (initial) + 2 training epochs
            self.assertEqual(len(history['auc']), 3)
            for v in history['train'] + history['val']:
                self.assertTrue(onp.isfinite(v))

            checkpoint_dir = os.path.join(save_dir, 'checkpoints')
            self.assertTrue(os.listdir(checkpoint_dir))  # per-epoch checkpoints were written
            trained_path = os.path.join(save_dir, 'trained_model.pickle')
            self.assertTrue(os.path.isfile(trained_path))

    def test_checkpoint_round_trip_and_resume_training(self) -> None:
        """A real saved checkpoint must reload and support one more training step."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=1)
            train_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)
            val_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)
            trainer.run_training_loop(train_loader, val_loader)

            trained_path = os.path.join(save_dir, 'trained_model.pickle')
            vqc2 = arch.QuantumClassifier(
                wires=4, shots=None, dev_name='default.qubit',
                layers=1, backend_name='autograd', test=False,
            )
            vqc2.set_circuit('normal', operations_per_qubit=3)
            vqc2.load_weights(trained_path, train=True)

            self.assertIsInstance(vqc2.current_weights, CircuitWeights)
            self.assertEqual(vqc2.current_weights.rot.shape, (1, 4, 3))

            # one more training step from the reloaded weights must not crash
            optimizer = qml.AdamOptimizer(stepsize=0.05)
            trainer2 = arch.QuantumTrainer(
                model=vqc2, lr=0.05, optimizer=optimizer, loss_fn=loss.VQC_cost,
                save=False, epochs=1, init_weights=vqc2.current_weights,
                batch_size=1, logger=None,
            )
            data = np.array(onp.random.uniform(-1, 1, size=(1, 4, 3)))
            labels = onp.array([1])
            cost = trainer2.iteration(data, labels=labels, train=True)
            self.assertTrue(onp.isfinite(cost))


class TestEvaluateEquivalentInference(unittest.TestCase):
    """D10: reproduce evaluate.py's own pipeline against a real saved checkpoint."""

    def test_run_inference_hits_missing_total_batches(self) -> None:
        """
        New finding: QuantumClassifier(test=False) -- exactly what evaluate.py and
        train.py both construct -- never sets self.total_batches (only the `if test:`
        branch of __init__ does). run_inference() unconditionally reads
        self.total_batches for tqdm's `total=`, so evaluate.py's real pipeline cannot
        work as written. Pre-existing, unrelated to this refactor -- not fixed.
        """
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=1)
            trainer.run_training_loop(
                _make_batches(2, 1, 4, 1), _make_batches(2, 1, 4, 1)
            )
            trained_path = os.path.join(save_dir, 'trained_model.pickle')

            vqc2 = arch.QuantumClassifier(
                wires=4, shots=None, dev_name='default.qubit',
                layers=1, backend_name='autograd', test=False,
            )
            vqc2.set_circuit('normal', operations_per_qubit=3)
            vqc2.load_weights(trained_path)
            self.assertFalse(hasattr(vqc2, 'total_batches'))

            test_loader = _make_batches(2, 1, 4, 1)
            with self.assertRaises(AttributeError):
                vqc2.run_inference(test_loader, loss_fn=loss.VQC_cost, loss_type='MSE')

    def test_run_inference_otherwise_works(self) -> None:
        """With total_batches worked around, the rest of the inference pipeline is sound."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=1)
            trainer.run_training_loop(
                _make_batches(2, 1, 4, 1), _make_batches(2, 1, 4, 1)
            )
            trained_path = os.path.join(save_dir, 'trained_model.pickle')

            vqc2 = arch.QuantumClassifier(
                wires=4, shots=None, dev_name='default.qubit',
                layers=1, backend_name='autograd', test=False,
            )
            vqc2.set_circuit('normal', operations_per_qubit=3)
            vqc2.load_weights(trained_path)
            vqc2.total_batches = 4  # workaround for the bug pinned down above

            test_loader = _make_batches(4, 1, 4, 1)
            costs, scores, labels = vqc2.run_inference(test_loader, loss_fn=loss.VQC_cost, loss_type='MSE')
            self.assertEqual(len(costs), 4)
            self.assertEqual(len(scores), 4)
            self.assertEqual(len(labels), 4)


class TestKnownPreExistingBugs(unittest.TestCase):
    """
    D10: pin down, with concrete reproductions, bugs already flagged (not fixed) in
    REFACTOR.md/AUDIT.md as pre-existing and out of this refactor's scope.
    """

    def test_validation_batch_size_above_one_crashes(self) -> None:
        """QuantumTrainer.iteration()'s validation branch only works at batch_size=1."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=1)
            train_loader = _make_batches(2, batch_size=1, n_qubits=4, n_layers=1)
            val_loader_batch2 = _make_batches(2, batch_size=2, n_qubits=4, n_layers=1)
            with self.assertRaises(TypeError):
                trainer.run_training_loop(train_loader, val_loader_batch2)

    def test_resume_path_passes_raw_dict_not_circuitweights(self) -> None:
        """train.py's resume branch does ut.Unpickle(model_path) without ['weights'],
        so init_weights ends up as the raw checkpoint dict, not a CircuitWeights."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=1)
            trainer.run_training_loop(_make_batches(2, 1, 4, 1), _make_batches(2, 1, 4, 1))
            trained_path = os.path.join(save_dir, 'trained_model.pickle')

            resumed_init_weights = ut.Unpickle(trained_path)  # mirrors train.py:56 exactly
            self.assertIsInstance(resumed_init_weights, dict)  # not a CircuitWeights

            optimizer = qml.AdamOptimizer(stepsize=0.05)
            trainer2 = arch.QuantumTrainer(
                model=vqc, lr=0.05, optimizer=optimizer, loss_fn=loss.VQC_cost,
                save=False, epochs=1, init_weights=resumed_init_weights,
                batch_size=1, logger=None,
            )
            data = np.array(onp.random.uniform(-1, 1, size=(1, 4, 3)))
            with self.assertRaises(AttributeError):
                trainer2.iteration(data, labels=onp.array([1]), train=True)

    def test_probabilistic_loss_incompatible_with_vqc_circuit(self) -> None:
        """probabilistic_loss expects a 2D (batch x class) circuit output; VQC's
        circuit returns one scalar Hamiltonian expval per sample -- mismatched shape."""
        vqc = arch.QuantumClassifier(
            wires=4, shots=None, dev_name='default.qubit',
            layers=1, backend_name='autograd', test=False,
        )
        vqc.set_circuit('normal', operations_per_qubit=3)
        shape = vqc._impl.rotation_shape(4, 1)
        weights = CircuitWeights(
            rot=np.array(onp.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R))),
            aux={k: np.array(v) for k, v in vqc._impl.aux_defaults.items()},
        )
        data = np.array(onp.random.uniform(-1, 1, size=(2, 4, 3)))
        with self.assertRaises(IndexError):
            loss.probabilistic_loss(
                weights, inputs=data, quantum_circuit=vqc.circuit,
                labels=onp.array([0, 1]), return_scores=True,
            )


if __name__ == '__main__':
    unittest.main()
