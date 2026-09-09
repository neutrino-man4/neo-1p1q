"""
Evaluate final trained weights using only their saved run configuration.
Locate config.yaml with --config PATH or --seed RUN [--model-dir DIRECTORY].
Author: Aritra Bal (ETP)
Date: 2026-09-09
"""
import glob
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import roc_curve, roc_auc_score

import case_reader as cr
import helpers.utils as ut
import quantum.losses as loss
from helpers.config import evaluation_config_path
from helpers.trained_run import load_trained_run


def main(config_path: str) -> None:
    """Verify a saved run, evaluate its final weights, and report test ROC/AUC."""
    config_path = str(pathlib.Path(config_path).expanduser().resolve())
    cfg, VQC = load_trained_run(config_path)
    save_dir = os.path.dirname(config_path)
    dump_dir = os.path.join(cfg.dump, cfg.seed)
    plot_dir = os.path.join(save_dir, 'plots')
    pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    logger.add(os.path.join(save_dir, 'eval.log'), rotation='10 MB', level='DEBUG')

    logger.info(f"Loaded verified final weights from {os.path.join(save_dir, 'trained_model.pickle')}")

    cost_fn = loss.VQC_cost

    test_split = 'flat_test' if cfg.flat else 'test'
    _class_files = lambda sample: sorted(glob.glob(os.path.join(cfg.data_dir, test_split, sample, '*.h5')))
    test_sig, test_bg = _class_files(cfg.signal), _class_files(cfg.background)
    if not (test_sig and test_bg):
        raise FileNotFoundError(
            f"Missing JetClass files under {cfg.data_dir} for signal='{cfg.signal}', "
            f"background='{cfg.background}' (split '{test_split}')"
        )
    logger.info(f"Test set: {cfg.n_signal_test} '{cfg.signal}' + {cfg.n_background_test} '{cfg.background}' jets from {test_split}/")

    required_particles = len(VQC.auto_wires) * VQC.num_layers
    num_particles = getattr(cfg, 'num_particles', required_particles)
    if num_particles < required_particles:
        raise ValueError(
            f"The {cfg.num_layers}-layer circuit requires at least "
            f"{required_particles} particles, but num_particles={num_particles}"
        )

    test_loader = cr.OneP1QDataLoader(
        signal_filelist=test_sig, background_filelist=test_bg,
        n_signal=cfg.n_signal_test, n_background=cfg.n_background_test,
        input_shape=(num_particles, 3),
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
    main(evaluation_config_path())
