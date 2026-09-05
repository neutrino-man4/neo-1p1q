"""
Run inference with a trained QuantumClassifier and report ROC/AUC on the held-out
test set. Replaces the old test.py / test_copy.py / test_jetclass.py, which targeted
a QuantumAutoencoder API that no longer exists. Point --config at the run's saved
config.yaml (or a base config with seed=<run> on the CLI) to pick which weights to
load.
Author: Aritra Bal (ETP)
Date: 2026-09-04
"""
import glob
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import roc_curve, roc_auc_score

import case_reader as cr
import helpers.utils as ut
import quantum.architectures as qc
import quantum.losses as loss
from helpers.config import load_config


def main(cfg: DictConfig) -> None:
    save_dir = os.path.join(cfg.save_dir, cfg.seed)
    dump_dir = os.path.join(cfg.dump, cfg.seed)
    plot_dir = os.path.join(save_dir, 'plots')
    pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    logger.add(os.path.join(save_dir, 'eval.log'), rotation='10 MB', level='DEBUG')

    # Rebuild the circuit with the same shape it was trained with
    VQC = qc.QuantumClassifier(
        wires=cfg.wires,
        shots=cfg.shots if cfg.shots > 0 else None,
        dev_name=cfg.device_name,
        layers=cfg.num_layers,
        backend_name=cfg.backend,
        test=False
    )
    VQC.set_circuit(circuit_type=cfg.circuit_type)

    # Prefer the final model; fall back to the latest checkpoint
    model_path = os.path.join(save_dir, 'trained_model.pickle')
    if not os.path.isfile(model_path):
        checkpoints = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))
        if not checkpoints:
            raise FileNotFoundError(f"No trained_model.pickle or checkpoints found under {save_dir}")
        model_path = checkpoints[-1]
        logger.warning(f"trained_model.pickle not found, using latest checkpoint: {model_path}")
    VQC.load_weights(model_path)
    logger.info(f"Loaded weights from {model_path}")

    cost_fn = loss.probabilistic_loss if cfg.loss == 'prob' else loss.VQC_cost

    test_split = 'flat_test' if cfg.flat else 'test'
    _class_files = lambda sample: sorted(glob.glob(os.path.join(cfg.data_dir, test_split, sample, '*.h5')))
    test_sig, test_bg = _class_files(cfg.signal), _class_files(cfg.background)
    if not (test_sig and test_bg):
        raise FileNotFoundError(
            f"Missing JetClass files under {cfg.data_dir} for signal='{cfg.signal}', "
            f"background='{cfg.background}' (split '{test_split}')"
        )
    logger.info(f"Test set: {cfg.n_signal_test} '{cfg.signal}' + {cfg.n_background_test} '{cfg.background}' jets from {test_split}/")

    test_loader = cr.OneP1QDataLoader(
        signal_filelist=test_sig, background_filelist=test_bg,
        n_signal=cfg.n_signal_test, n_background=cfg.n_background_test,
        input_shape=(len(VQC.auto_wires), 3),
        train=False,
        normalize_pt=cfg.norm_pt,
        logger=logger,
    )

    costs, scores, labels = VQC.run_inference(test_loader, loss_fn=cost_fn, loss_type=cfg.loss)

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    logger.info(f"Test AUC: {auc:.4f}")

    ut.Pickle({'costs': costs, 'scores': scores, 'labels': labels, 'auc': auc}, 'test_results', path=dump_dir)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate', size=16)
    ax.set_ylabel('True Positive Rate', size=16)
    ax.legend(prop={'size': 14})
    fig.savefig(os.path.join(plot_dir, 'roc_curve.png'))
    logger.info(f"Saved ROC curve to {os.path.join(plot_dir, 'roc_curve.png')}")


if __name__ == "__main__":
    main(load_config())
