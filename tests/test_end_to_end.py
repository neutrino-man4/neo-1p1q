"""
D10: end-to-end validation. train.py itself can't run (its pre-existing,
deliberately-unfixed config_name bug, see AUDIT.md), so this exercises the
same pieces train.py/evaluate.py wire together -- QuantumTrainer's full
run_training_loop, a real checkpoint save/load round trip, and the
evaluate.py-equivalent inference pipeline -- built directly, without hydra.

Includes regression tests for the training, validation, and inference paths.
Author: Aritra Bal (ETP)
2026-09-05
"""
import os
import tempfile
import unittest

import numpy as onp
import pennylane as qml
import pennylane.numpy as np

import glob

import case_reader as cr
import helpers.utils as ut
import quantum.architectures as arch
import quantum.losses as loss
from quantum.circuits.base import CircuitWeights

_JETCLASS_DIR = '/ceph/abal/JetClass'


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

    def test_batched_training_and_validation_include_partial_batch(self) -> None:
        """Batch scores retain their dimension and epoch losses weight every sample equally."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=2, epochs=1)
            data_batches = [
                np.array(onp.random.uniform(-1, 1, size=(2, 4, 3))),
                np.array(onp.random.uniform(-1, 1, size=(1, 4, 3))),
            ]
            label_batches = [np.array([0, 1]), np.array([1])]
            train_loader = _FixedBatches(data_batches, label_batches)
            val_loader = _FixedBatches(data_batches, label_batches)

            batch_losses = []
            for data, labels in val_loader:
                batch_loss, scores = trainer.iteration(data, labels=labels, train=False)
                batch_losses.append(batch_loss)
                self.assertEqual(scores.shape, (len(labels),))
                self.assertAlmostEqual(
                    batch_loss,
                    float(np.mean((labels - scores) ** 2)),
                )
                serial_scores = np.concatenate([
                    trainer.iteration(
                        data[i:i + 1], labels=labels[i:i + 1], train=False
                    )[1]
                    for i in range(len(labels))
                ])
                onp.testing.assert_allclose(scores, serial_scores)
            expected_initial_val_loss = (
                batch_losses[0] * 2 + batch_losses[1]
            ) / 3

            history = trainer.run_training_loop(train_loader, val_loader)

            self.assertAlmostEqual(history['val'][0], expected_initial_val_loss)
            self.assertEqual(len(history['train']), 1)
            self.assertEqual(len(history['val']), 2)
            self.assertTrue(onp.isfinite(history['train'][0]))


class TestEvaluateEquivalentInference(unittest.TestCase):
    """D10: reproduce evaluate.py's own pipeline against a real saved checkpoint."""

    def _trained_classifier(self, save_dir: str):
        """Train a tiny model and reload it the way evaluate.py does (test=False)."""
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
        return vqc2

    def test_run_inference_runs_without_total_batches(self) -> None:
        """
        P2 fix: QuantumClassifier(test=False) -- what evaluate.py/train.py build --
        no longer needs self.total_batches. run_inference() just walks the loader,
        so evaluate.py's real pipeline runs end to end.
        """
        with tempfile.TemporaryDirectory() as save_dir:
            vqc2 = self._trained_classifier(save_dir)
            self.assertFalse(hasattr(vqc2, 'total_batches'))

            test_loader = _make_batches(4, 1, 4, 1)
            costs, scores, labels = vqc2.run_inference(test_loader, loss_fn=loss.VQC_cost, loss_type='MSE')
            self.assertEqual(len(costs), 4)
            self.assertEqual(len(scores), 4)
            self.assertEqual(len(labels), 4)

    def test_run_inference_processes_multi_jet_chunks_one_at_a_time(self) -> None:
        """A loader that groups jets is still scored jet by jet: N jets in -> N results out."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc2 = self._trained_classifier(save_dir)

            test_loader = _make_batches(3, 4, 4, 1)  # 3 chunks x 4 jets = 12 jets
            costs, scores, labels = vqc2.run_inference(test_loader, loss_fn=loss.VQC_cost, loss_type='MSE')
            self.assertEqual(len(costs), 12)
            self.assertEqual(len(scores), 12)
            self.assertEqual(len(labels), 12)


class TestKnownPreExistingBugs(unittest.TestCase):
    """
    D10: pin down, with concrete reproductions, bugs already flagged (not fixed) in
    REFACTOR.md/AUDIT.md as pre-existing and out of this refactor's scope.
    """

    def test_trainer_rejects_raw_checkpoint_dictionary(self) -> None:
        """Trainer callers must extract the weights from a checkpoint dictionary."""
        with tempfile.TemporaryDirectory() as save_dir:
            vqc, trainer = _build_trainer(save_dir, batch_size=1, epochs=1)
            trainer.run_training_loop(_make_batches(2, 1, 4, 1), _make_batches(2, 1, 4, 1))
            trained_path = os.path.join(save_dir, 'trained_model.pickle')

            resumed_init_weights = ut.Unpickle(trained_path)
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


@unittest.skipUnless(
    os.path.isdir(os.path.join(_JETCLASS_DIR, 'test', 'TTBar_'))
    and os.path.isdir(os.path.join(_JETCLASS_DIR, 'test', 'ZJetsToNuNu')),
    "JetClass data not available",
)
class TestBalancedJetClassLoader(unittest.TestCase):
    """CASEJetClassDataset must read equal signal/background and label them correctly."""

    def _files(self, sample):
        return sorted(glob.glob(os.path.join(_JETCLASS_DIR, 'test', sample, '*.h5')))

    def test_equal_split_and_both_labels(self) -> None:
        loader = cr.OneP1QDataLoader(
            signal_filelist=self._files('TTBar_'),
            background_filelist=self._files('ZJetsToNuNu'),
            n_signal=40, n_background=40,
            batch_size=18, input_shape=(4, 3), train=True, normalize_pt=False, seed=0,
        )
        total, counts = 0, {0: 0, 1: 0}
        first_shape = None
        batch_sizes = []
        for data, labels in loader:
            if first_shape is None:
                first_shape = tuple(data.shape[1:])
            batch_sizes.append(data.shape[0])
            for lbl in labels.tolist():
                counts[int(lbl)] += 1
            total += data.shape[0]
        self.assertEqual(total, 80)
        self.assertEqual(counts, {0: 40, 1: 40})
        self.assertEqual(first_shape, (4, 3))
        self.assertEqual(len(loader), 5)
        self.assertEqual(batch_sizes, [18, 18, 18, 18, 8])

    def test_signal_label_is_one(self) -> None:
        """A signal-only loader yields only label 1; background-only yields only label 0."""
        sig = cr.OneP1QDataLoader(
            signal_filelist=self._files('TTBar_'), background_filelist=self._files('TTBar_'),
            n_signal=20, n_background=0, batch_size=20, input_shape=(4, 3), seed=0,
        )
        labels = [int(x) for _, lbls in sig for x in lbls.tolist()]
        self.assertEqual(set(labels), {1})
        self.assertEqual(len(labels), 20)


if __name__ == '__main__':
    unittest.main()
