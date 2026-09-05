"""
Run inference with a trained QuantumClassifier and report ROC/AUC on the held-out
test set. Replaces the old test.py / test_copy.py / test_jetclass.py, which targeted
a QuantumAutoencoder API that no longer exists. Mirrors train.py's Hydra config
style: point --seed at the run whose weights you want to load.
Author: Aritra Bal (ETP)
Date: 2026-09-04
"""
import glob
import os
import pathlib

import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import roc_curve, roc_auc_score

import case_reader as cr
import helpers.path_setter as ps
import helpers.utils as ut
import quantum.architectures as qc
import quantum.losses as loss


@hydra.main(config_path="./hydra_configs/VQC", config_name="base")
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

    test_key = 'VQC_test'
    test_path = ps.PathSetter(data_path=cfg.data_dir).get_data_path(test_key)
    # JetClass files sit one level down per sample type; interleave them by file number.
    test_filelist = sorted(
        glob.glob(os.path.join(test_path, '**', '*.h5'), recursive=True),
        key=lambda p: os.path.basename(p).rsplit('_', 1)[-1],
    )
    if len(test_filelist) == 0:
        raise FileNotFoundError(f"Could not find test files at {test_path}")
    logger.info(f"Testing on {len(test_filelist)} files found at {test_path}")

    test_loader = cr.OneP1QDataLoader(
        filelist=test_filelist,
        batch_size=1,               # run_inference expects batch_size=1
        input_shape=(len(VQC.auto_wires), 3),
        train=False,
        max_samples=cfg.read_n,
        normalize_pt=cfg.norm_pt,
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
    main()
