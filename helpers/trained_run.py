"""
Save and verify the configuration, implementation, and final weights of a run.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""

import hashlib
from importlib.metadata import distributions
from pathlib import Path
import pickle
from typing import Any
import warnings

import numpy as np
import pennylane.numpy as qnp
from omegaconf import DictConfig, OmegaConf

from quantum.architectures import QuantumClassifier
from quantum.circuits.base import CircuitWeights


def implementation_signature() -> dict[str, Any]:
    """Fingerprint local model/data code and the numerical library versions."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'quantum').rglob('*.py'))
    paths += [root / 'helpers/utils.py', root / 'case_reader.py']
    packages = {}
    for distribution in distributions():
        name = distribution.metadata['Name'].lower().replace('_', '-')
        if name.startswith('pennylane') or name in ('numpy', 'autograd', 'scipy'):
            packages[name] = distribution.version
    return {
        'sources': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in paths},
        'packages': packages,
    }


def validate_weights(model: QuantumClassifier, weights: Any, cfg: DictConfig) -> None:
    """Reject missing, nonfinite, or incompatible structured circuit weights."""
    if not isinstance(weights, CircuitWeights):
        raise ValueError('The trained model must contain structured CircuitWeights.')
    shape = model._impl.rotation_shape(len(model.auto_wires), model.num_layers)
    expected_shape = (shape.L, shape.N, shape.R)
    if np.shape(weights.rot) != expected_shape:
        raise ValueError(f'Rotation weights have shape {np.shape(weights.rot)}; expected {expected_shape}.')
    auxiliary = {**model._impl.aux_defaults, **dict(cfg.get('aux_weights', {}))}
    if set(weights.aux) != set(auxiliary):
        raise ValueError('Auxiliary weight names do not match the saved configuration.')
    for name, initial in auxiliary.items():
        if np.shape(weights.aux[name]) != np.shape(initial):
            raise ValueError(f'Auxiliary weight shape does not match the configuration: {name}')
    if not all(np.all(np.isfinite(value)) for value in [weights.rot, *weights.aux.values()]):
        raise ValueError('Trained weights contain nonfinite values.')


def save_trained_run(
    run_dir: str, cfg: DictConfig, model: QuantumClassifier, weights: CircuitWeights,
    history: dict[str, list[float]], signature: dict[str, Any],
) -> None:
    """Write final weights with provenance only after successful training."""
    epochs = len(history['train'])
    if (epochs < 1 or len(history['val']) != epochs + 1 or len(history['auc']) != epochs + 1
            or not all(np.all(np.isfinite(history[key])) for key in ('train', 'val', 'auc'))):
        raise ValueError('Cannot certify final weights: incomplete training history or nonfinite metrics.')
    if implementation_signature() != signature:
        raise ValueError('Model implementation changed during training; final weights cannot be verified.')
    validate_weights(model, weights, cfg)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    saved_config = OmegaConf.to_container(OmegaConf.load(Path(run_dir) / 'config.yaml'), resolve=True)
    if config != saved_config:
        raise ValueError('Saved YAML differs from the configuration used for training.')
    payload = {
        'weights': weights,
        'training': {
            'config': config,
            'implementation': signature,
            'completed_epochs': epochs,
            'stop_reason': 'early_stopping' if epochs < cfg.epochs else 'epoch_limit',
        },
    }
    path = Path(run_dir) / 'trained_model.pickle'
    temporary = path.with_suffix('.pickle.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(payload, stream)
    temporary.replace(path)


def load_trained_run(config_path: str) -> tuple[DictConfig, QuantumClassifier]:
    """Load the final model beside its YAML, rejecting unverifiable run artifacts."""
    path = Path(config_path).expanduser().resolve()
    cfg = OmegaConf.load(path)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    model_path = path.parent / 'trained_model.pickle'
    if not model_path.is_file():
        raise FileNotFoundError(f'Missing final trained weights: {model_path}; intermediate checkpoints are not accepted.')
    with model_path.open('rb') as stream:
        payload = pickle.load(stream)
    training = payload.get('training') if isinstance(payload, dict) else None
    if not isinstance(training, dict) or training.get('completed_epochs', 0) < 1:
        raise ValueError('Missing completed-training provenance; cannot verify these final weights.')
    if training.get('config') != config:
        raise ValueError('Saved YAML does not match the configuration recorded with the trained weights.')
    if training.get('implementation') != implementation_signature():
        raise ValueError('Circuit/data source or numerical library versions differ from training.')
    model = QuantumClassifier.from_config(cfg)
    weights = payload.get('weights')
    validate_weights(model, weights, cfg)
    model.current_weights = CircuitWeights(
        rot=qnp.array(weights.rot, requires_grad=False),
        aux={name: qnp.array(value, requires_grad=False) for name, value in weights.aux.items()},
    )
    warnings.warn(
        f"Loaded final weights after {training['completed_epochs']} epochs "
        f"({training.get('stop_reason', 'unknown stop reason')}). "
        'Training completion does not establish convergence; inspect the validation history.',
        RuntimeWarning,
        stacklevel=2,
    )
    return cfg, model
