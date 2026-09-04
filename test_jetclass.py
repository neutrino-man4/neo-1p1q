import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import glob, h5py, sys, json
# Other imports
import numpy as nnp
import helpers.utils as ut
import helpers.path_setter as ps
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mplhep; mplhep.style.use("CMS")
from sklearn.metrics import roc_curve, roc_auc_score
from plotters.plot_utils import plot_qfi_matrices

@hydra.main(config_path="./hydra_configs/VQC", config_name="config")
def main(cfg: DictConfig):
    log_wandb = cfg.log_wandb
    # Set up directories
    save_dir = os.path.join(cfg.save_dir, cfg.seed)
    out_dir = os.path.join(cfg.dump, cfg.seed)
    plot_dir = os.path.join(save_dir, 'plots')

    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
    
    if log_wandb:
        run_id_path = os.path.join(save_dir, "wandb_run_id.txt")
        try:
            assert os.path.exists(run_id_path), "wandb_run_id.txt not found. Ensure training script saved the run ID."
            import wandb
            with open(run_id_path, "r") as f:
                run_id = f.read().strip()

            # Reuse the existing WandB run
            wandb.init(project='1P1Q', id=run_id, resume="must", config=OmegaConf.to_container(cfg))
        except:
            print("WandB run not found. Logging disabled.")
            log_wandb = False
    
    # Import frozen architecture if available
    try:
        import importlib.util
        qc_spec = importlib.util.spec_from_file_location('qc', os.path.join(save_dir, 'FROZEN_ARCHITECTURE.py'))
        qc = importlib.util.module_from_spec(qc_spec)
        qc_spec.loader.exec_module(qc)
        
        cr_spec = importlib.util.spec_from_file_location('cr', os.path.join(save_dir, 'FROZEN_DATAREADER.py'))
        cr = importlib.util.module_from_spec(cr_spec)
        cr_spec.loader.exec_module(cr)
        
        loss_spec = importlib.util.spec_from_file_location('loss', os.path.join(save_dir, 'FROZEN_LOSS.py'))
        loss = importlib.util.module_from_spec(loss_spec)
        loss_spec.loader.exec_module(loss)
        print("Successfully imported frozen architecture and dataloaders")
    except ImportError as e:
        print(e)
        import quantum.architectures as qc
        import case_reader as cr
        import quantum.losses as loss
        print("Failure: Frozen architecture not imported. Fetching generic architecture instead")
        sys.exit(0)
        
    # Load arguments and set up quantum classifier
    norm_pt = cfg.norm_pt
    if cfg.compute_fisher:
        if cfg.metric_approximation=='none':
                cfg.metric_approximation = None
        if cfg.metric_approximation=='adjoint':
            print("Using adjoint method for Fisher computation. Requires shots=None (analytic computation) and default.qubit device")
            cfg.device_name = 'default.qubit'
            cfg.shots= None

    qClassifier = qc.QuantumClassifier(
        wires=cfg.wires, 
        dev_name=cfg.device_name,shots=cfg.shots, 
        layers=cfg.num_layers,
        backend_name=cfg.backend,approx=cfg.metric_approximation,
        test=True,fisher_computation=cfg.compute_fisher,read_n=cfg.read_n,
    )
    qClassifier.set_circuit(circuit_type=cfg.circuit_type)
    qClassifier.print_training_params()
    cost_fn = loss.VQC_cost
    
    # Load model weights
    try:
        model_path = os.path.join(save_dir, 'trained_model.pickle')
        assert os.path.isfile(model_path), 'Model not found at: ' + model_path
    except:
        model_path = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[-1]
        print(f"Trained model not found. Will load the last checkpoint instead: {model_path}")
    
    if cfg.load_best:
        history = ut.Unpickle(os.path.join(save_dir, 'history.pickle'))
        val_auc = history['auc']
        best_model = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[nnp.argmax(val_auc)-1]
        model_path = best_model
        print(f"Loading best model based on validation AUC: {model_path}")
    
    qClassifier.load_weights(model_path,train=True,rearrange=False)
    print(f"Successfully loaded model at {model_path}")

    # Load test data with unified DataLoader approach
    paths = ps.PathSetter(data_path=cfg.data_dir)
    test_filelist = sorted(glob.glob(paths.get_data_path('VQC_test') + '/*.h5'))
    
    # Create DataLoader with batch_size=1 for both inference and Fisher computation
    num_particles = getattr(cfg, 'num_particles', len(qClassifier.auto_wires))
    
    test_loader = cr.OneP1QDataLoader(
        filelist=test_filelist,
        batch_size=1,  # Single sample per batch for simplified processing
        input_shape=(num_particles, 3),
        train=False,
        max_samples=cfg.read_n,
        normalize_pt=norm_pt,
        dataset=cfg.dataset
    )
    
    if cfg.load:
        with h5py.File(os.path.join(out_dir, 'qcd_results.h5'), 'r') as f:
            scores = f['scores'][()]
            costs = f['costs'][()]
            labels = f['truth_labels'][()]
            try:
                fisher_matrices = f['fisher_matrices'][()]
            except KeyError:
                print("Fisher matrices not found in the file.")
                fisher_matrices = nnp.zeros((10, 10, 10))  # Placeholder for Fisher matrices if not found
    else:
        # Run inference using the same dataloader
        print("Running inference...")
        costs, scores, labels = qClassifier.run_inference(
            dataloader=test_loader,
            loss_fn=cost_fn,
            loss_type=cfg.loss
        )
        
        # Calculate Fisher Information Matrices using the same dataloader
        print("Computing Fisher Information Matrices...")
        if cfg.compute_fisher:
            fisher_matrices, labels = qClassifier.run_fisher_computation(test_loader)
            print(f"Fisher matrices shape: {fisher_matrices.shape}")
        
        # Save results
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(os.path.join(out_dir, 'qcd_results.h5'), 'w') as f:
            f.create_dataset('scores', data=scores)
            f.create_dataset('costs', data=costs)
            f.create_dataset('truth_labels', data=labels)
            if cfg.compute_fisher:
                f.create_dataset('fisher_matrices', data=fisher_matrices)
        
        print(f"Results saved to {out_dir}")
        print(f"Processed {len(scores)} samples")
        
    # Calculate ROC metrics
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    print(f'AUC={roc_auc:.3f}')
    #import pdb; pdb.set_trace()
    ci = nnp.argmin(nnp.abs(tpr - 0.99))
    rej_99 = 1 / fpr[ci]
    print(f"Background rejection at 99% signal efficiency: {rej_99:.3f}")
    
    # Save ROC data
    plot_label = r'$t \rightarrow bq\overline{q}$ jets'
    pathlib.Path(os.path.join(plot_dir, 'ROC_data')).mkdir(parents=True, exist_ok=True)
    npz_path = os.path.join(plot_dir, 'ROC_data', 'FPR_TPR.npz')
    nnp.savez(npz_path, fpr=fpr, tpr=tpr, auc=roc_auc, thresholds=thresholds, rej_99=rej_99)
    print(f"ROC curve data saved to {npz_path}")
    
    # Save metrics as json
    with open(os.path.join(plot_dir, 'ROC_data', 'metrics.json'), 'w') as f:
        json.dump({'auc': roc_auc, 'rej_99': rej_99}, f)
    
    # Plot classifier score histogram
    bins_qcd, edges_qcd = nnp.histogram(scores[labels == 0], density=True, bins=50, range=[0, 2])
    bins_sig, edges_sig = nnp.histogram(scores[labels == 1], density=True, bins=50, range=[0, 2])
    plt.stairs(bins_qcd, edges_qcd, fill=True, label='q/g jets', alpha=0.6)
    plt.stairs(bins_sig, edges_sig, fill=False, label=plot_label)
    plt.minorticks_on()
    plt.xlabel('Classifier Score')
    plt.ylabel('No. of events')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plot_dir, f'classifier_score_hist.png'))

    # Plot ROC curve
    plt.clf()
    plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {plot_label} vs q/g jets')
    plt.minorticks_on()
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(plot_dir, f'roc_curve.png'))
    
    # Plot SIC
    plt.clf()
    sic = tpr / nnp.sqrt(fpr)
    plt.plot(tpr, sic)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Significance Improvement')
    plt.xlim(0.1, 1)
    plt.title(f'SIC: {plot_label} vs q/g jets')
    plt.savefig(os.path.join(plot_dir, f'SIC.png'))
    
    # Plot rejection
    plt.clf()
    plt.plot(tpr, 1./fpr, label='Rej$_{X}$')
    plt.xlabel('TPR')
    plt.ylabel('1/FPR')
    plt.yscale('log')
    plt.title(f'Rejection vs Signal Efficiency: {plot_label} vs q/g jets')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'rejection.png'))
    plot_qfi_matrices(fisher_matrices[labels==1], plot_label=r'$t \to bq \overline{q^\prime}$',save_path=os.path.join(plot_dir, 'qfi_matrices/top.pdf'))
    plot_qfi_matrices(fisher_matrices[labels==0], plot_label=r'$q/g$ jets',save_path=os.path.join(plot_dir, 'qfi_matrices/qcd.pdf'))
    
    # Log to WandB if enabled
    if log_wandb:
        for filename in glob.glob(os.path.join(plot_dir, "*.png")):
            wandb.log({os.path.split(filename)[-1].replace('.png', ''): wandb.Image(filename)})
        wandb.log({
            'test_AUC': roc_auc,
            'background_rejection_99': rej_99,
            'fisher_matrix_shape': fisher_matrices.shape,
            'num_test_samples': len(scores)
        })
        wandb.finish()



# def plot_qfi_matrices(qfi_matrices, labels, scores, save_path=None):
#     """
#     Plot averaged QFI matrices for each label class.
    
#     Args:
#         qfi_matrices: Array of shape (N, N_input, N_input) - QFI matrices for N samples
#         labels: Array of shape (N,) - corresponding labels (0 or 1)
#         save_path: Optional path to save the plots
#     """
#     # Separate matrices by label
#     label_0_mask = (labels == 0)
#     label_1_mask = (labels == 1)
#     label_0_scores=scores[label_0_mask]
#     label_1_scores=scores[label_1_mask]
#     qfi_label_0 = qfi_matrices[label_0_mask]  # Shape: (N_0, N_input, N_input)
#     qfi_label_1 = qfi_matrices[label_1_mask]  # Shape: (N_1, N_input, N_input)
    
#     # Average over samples for each label
#     #avg_qfi_0 = nnp.mean(qfi_label_0, axis=0)  # Shape: (N_input, N_input)
#     #avg_qfi_1 = nnp.mean(qfi_label_1, axis=0)  # Shape: (N_input, N_input)
#     pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        
#     max_label_1= nnp.argmax(label_1_scores)
#     min_label_0= nnp.argmin(label_0_scores)
#     #for i in range(100):

#     avg_qfi_0=nnp.mean(qfi_label_0, axis=0)  # Shape: (N_input, N_input)
#     avg_qfi_1=nnp.mean(qfi_label_1, axis=0)  # Shape: (N_input, N_input)
#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Use symmetric log scaling to handle values near zero and both positive/negative
    
#     # Determine appropriate scaling threshold (1% of max absolute value)
#     #import pdb;pdb.set_trace()
#     # Plot for label 0
#     colors = ['#0066FF', 'white', '#FF0066']  # Blue -> White -> Red
#     cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
#     #import pdb;pdb.set_trace()
#     # Set up symmetric logarithmic normalization
#     linthresh = 1.0e-4  # Linear threshold - values between -0.01 and 0.01 will be linear
#     norm = matplotlib.colors.Normalize(vmin=-0.25, vmax=0.25)

#     im1 = ax1.matshow(avg_qfi_0, cmap=cmap, norm=norm)
#                     #norm=SymLogNorm(linthresh=linthresh, vmin=-max_abs_val, vmax=max_abs_val))
#     #ax1.set_title(f'Average QFI Matrix - Label 0\n({nnp.sum(label_0_mask)} samples)')
#     ax1.set_xlabel('Parameter Index')
#     ax1.set_ylabel('Parameter Index')
#     cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
#     cbar1.set_label('QFI Value')
    
#     # Plot for label 1
#     im2 = ax2.matshow(avg_qfi_1, cmap=cmap, norm=norm)
#                     #norm=SymLogNorm(linthresh=linthresh, vmin=-max_abs_val, vmax=max_abs_val))
#     #ax2.set_title(f'Average QFI Matrix - Label 1\n({nnp.sum(label_1_mask)} samples)')
#     ax2.set_xlabel('Parameter Index')
#     ax2.set_ylabel('Parameter Index')
#     cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
#     cbar2.set_label('QFI Value')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path,f'average_QFI.pdf'), dpi=300, bbox_inches='tight')
#     plt.clf()

#     # Print some statistics
#     print(f"Label 0 - Samples: {nnp.sum(label_0_mask)}, Avg QFI Range: [{avg_qfi_0.min():.4f}, {avg_qfi_0.max():.4f}]")
#     print(f"Label 1 - Samples: {nnp.sum(label_1_mask)}, Avg QFI Range: [{avg_qfi_1.min():.4f}, {avg_qfi_1.max():.4f}]")
if __name__ == "__main__":
    main()