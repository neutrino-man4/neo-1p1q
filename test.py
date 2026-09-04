import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import glob

# Other imports
import quantum.losses as loss
import numpy as nnp
import helpers.utils as ut
import helpers.path_setter as ps
import case_reader as cr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

@hydra.main(config_path="./hydra_configs", config_name="config")
def main(cfg: DictConfig):
    scheme = cfg.scheme
    log_wandb=cfg.log_wandb

    # Set up directories
    save_dir = os.path.join(cfg.save_dir, cfg.seed)
    dump_dir = os.path.join(cfg.dump, cfg.seed)
    plot_dir = os.path.join(save_dir, 'plots')
    hist_dir = os.path.join(plot_dir, 'mjj_histograms')

    pathlib.Path(hist_dir).mkdir(parents=True, exist_ok=True)
    run_id_path = os.path.join(save_dir, "wandb_run_id.txt")
    if log_wandb:
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
        import case_reader as cr
        #cr_spec = importlib.util.spec_from_file_location('cr', os.path.join(save_dir, 'FROZEN_DATAREADER.py'))
        #cr = importlib.util.module_from_spec(cr_spec)
        #cr_spec.loader.exec_module(cr)
        
        #print("Successfully imported frozen architecture and dataloaders")
    except ImportError:
        import quantum.architectures as qc
        print("Failure: Frozen architecture not imported. Fetching generic architecture instead")

    # Load arguments and set up quantum autoencoder
    

    sep = cfg.separate_ancilla
    norm_pt = cfg.norm_pt

    qAE = qc.QuantumAutoencoder(wires=cfg.wires, trash_qubits=cfg.trash_qubits, dev_name=cfg.device_name, separate_ancilla=sep, test=True)
    qAE.set_circuit(reuploading=cfg.use_reuploading)
    qc.print_training_params()
    if cfg.loss=='quantum':
        cost_fn = loss.quantum_cost
    else:
        cost_fn = loss.semi_classical_cost
    

    # Load model weights
    try:
        model_path = os.path.join(save_dir, 'trained_model.pickle')
        assert os.path.isfile(model_path), 'Model not found at: ' + model_path
    except:
        print(f"Trained model not found at {model_path}. Will load the most recent checkpoint instead.")
        model_path = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[-1]
    finally:
        qAE.load_weights(model_path)
        print(f"Successfully loaded model at {model_path}")

    # Load test data
    paths = ps.PathSetter(data_path=cfg.data_dir)
    sig_files = sorted(glob.glob(paths.get_data_path(cfg.signal) + '/*.h5'))
    if cfg.dataset.casefold()=='delphes':
        qcd_files = sorted(glob.glob(paths.get_data_path('QCD_SR_test') + '/*.h5'))
        qcd_dataset=cr.CASEDelphesJetDataset(
            filelist=qcd_files,
            input_shape=(len(qc.auto_wires), 3),
            normalize_pt=norm_pt,
            max_samples=cfg.read_n,
            use_subjet_PFCands=cfg.substructure
        )
        sig_dataset=cr.CASEDelphesJetDataset(
            filelist=sig_files,
            input_shape=(len(qc.auto_wires), 3),
            normalize_pt=norm_pt,
            max_samples=cfg.read_n,
            use_subjet_PFCands=cfg.substructure
        )
    elif cfg.dataset.casefold()=='jetclass': 
        qcd_files=sorted(glob.glob(paths.get_data_path('ZJetsToNuNu_val') + '/*.h5'))
        qcd_dataset=cr.CASEJetClassDataset(
            filelist=qcd_files,
            input_shape=(len(qc.auto_wires), 3),
            normalize_pt=norm_pt,
            max_samples=cfg.read_n,
            use_subjet_PFCands=cfg.substructure,selection=cfg.PFCand_selection_type
        )
        sig_dataset=cr.CASEJetClassDataset(
            filelist=sig_files,
            input_shape=(len(qc.auto_wires), 3),
            normalize_pt=norm_pt,
            max_samples=cfg.read_n,
            use_subjet_PFCands=cfg.substructure,selection=cfg.PFCand_selection_type
        )
    else:
        raise NameError(f"Dataset {cfg.dataset} not recognized. Must be either delphes or jetclass")
    
    

    if cfg.load:
        qcd_fids_j1=nnp.load(os.path.join(dump_dir,'qcd_fids_j1.npy'))
        sig_fids_j1=nnp.load(os.path.join(dump_dir,cfg.signal,'sig_fids_j1.npy'))
        qcd_fids_j2=nnp.load(os.path.join(dump_dir,'qcd_fids_j2.npy'))
        sig_fids_j2=nnp.load(os.path.join(dump_dir,cfg.signal,'sig_fids_j2.npy'))
        qcd_costs_j1=nnp.load(os.path.join(dump_dir,'qcd_costs_j1.npy'))
        sig_costs_j1=nnp.load(os.path.join(dump_dir,cfg.signal,'sig_costs_j1.npy'))
        qcd_costs_j2=nnp.load(os.path.join(dump_dir,'qcd_costs_j2.npy'))
        sig_costs_j2=nnp.load(os.path.join(dump_dir,cfg.signal,'sig_costs_j2.npy'))
        qcd_features=nnp.load(os.path.join(dump_dir,'qcd_features.npy'))
        sig_features=nnp.load(os.path.join(dump_dir,cfg.signal,'sig_features.npy'))
        qcd_labels=nnp.zeros(qcd_fids_j1.shape[0])
        sig_labels=nnp.ones(sig_fids_j1.shape[0])
    # Inference
    else:
        qcd_j1_etaphipt, qcd_j2_etaphipt, qcd_features, qcd_labels = qcd_dataset.load_for_inference()
        sig_j1_etaphipt, sig_j2_etaphipt, sig_features, sig_labels = sig_dataset.load_for_inference()
    
        qcd_costs_j1, qcd_fids_j1 = qAE.run_inference(qcd_j1_etaphipt, loss_fn=cost_fn)
        qcd_costs_j2, qcd_fids_j2 = qAE.run_inference(qcd_j2_etaphipt, loss_fn=cost_fn)
        sig_costs_j1, sig_fids_j1 = qAE.run_inference(sig_j1_etaphipt, loss_fn=cost_fn)
        sig_costs_j2, sig_fids_j2 = qAE.run_inference(sig_j2_etaphipt, loss_fn=cost_fn)

        # Save results
        pathlib.Path(os.path.join(dump_dir, cfg.signal)).mkdir(parents=True, exist_ok=True)
        nnp.save(os.path.join(dump_dir, 'qcd_fids_j1.npy'), qcd_fids_j1)
        nnp.save(os.path.join(dump_dir, cfg.signal, 'sig_fids_j1.npy'), sig_fids_j1)
        nnp.save(os.path.join(dump_dir, 'qcd_fids_j2.npy'), qcd_fids_j2)
        nnp.save(os.path.join(dump_dir, cfg.signal, 'sig_fids_j2.npy'), sig_fids_j2)
        nnp.save(os.path.join(dump_dir, 'qcd_costs_j1.npy'), qcd_costs_j1)
        nnp.save(os.path.join(dump_dir, cfg.signal, 'sig_costs_j1.npy'), sig_costs_j1)
        nnp.save(os.path.join(dump_dir, 'qcd_costs_j2.npy'), qcd_costs_j2)
        nnp.save(os.path.join(dump_dir, cfg.signal, 'sig_costs_j2.npy'), sig_costs_j2)
        nnp.save(os.path.join(dump_dir, 'qcd_features.npy'), qcd_features)
        nnp.save(os.path.join(dump_dir, cfg.signal, 'sig_features.npy'), sig_features)

    # Plotting, ROC curve, and other analysis (the rest of your code goes here...)
    if scheme=='max':
        print("Using max (j1,j2) loss")
        plot_dir=os.path.join(plot_dir,'max_loss');pathlib.Path(plot_dir).mkdir(parents=True,exist_ok=True)
        qcd_fids=nnp.minimum(qcd_fids_j1,qcd_fids_j2)#0.5*(qcd_fids_j1+qcd_fids_j2)#nnp.maximum(qcd_fids_j1,qcd_fids_j2)
        sig_fids=nnp.minimum(sig_fids_j1,sig_fids_j2)#0.5*(sig_fids_j1+sig_fids_j2)#nnp.maximum(sig_fids_j1,sig_fids_j2)
        qcd_costs=nnp.maximum(qcd_costs_j1,qcd_costs_j2)#0.5*(qcd_costs_j1+qcd_costs_j2)#nnp.minimum(qcd_costs_j1,qcd_costs_j2)
        sig_costs=nnp.maximum(sig_costs_j1,sig_costs_j2)#0.5*(sig_costs_j1+sig_costs_j2)#nnp.minimum(sig_costs_j1,sig_costs_j2)
    elif scheme=='min':
        print("Using min (j1,j2) loss")
        plot_dir=os.path.join(plot_dir,'min_loss');pathlib.Path(plot_dir).mkdir(parents=True,exist_ok=True)
        qcd_fids=nnp.maximum(qcd_fids_j1,qcd_fids_j2)
        sig_fids=nnp.maximum(sig_fids_j1,sig_fids_j2)
        qcd_costs=nnp.minimum(qcd_costs_j1,qcd_costs_j2)
        sig_costs=nnp.minimum(sig_costs_j1,sig_costs_j2)
    elif scheme=='mean':
        print("Using avg (j1,j2) loss")
        plot_dir=os.path.join(plot_dir,'avg_loss');pathlib.Path(plot_dir).mkdir(parents=True,exist_ok=True)
        qcd_fids=0.5*(qcd_fids_j1+qcd_fids_j2)
        sig_fids=0.5*(sig_fids_j1+sig_fids_j2)
        qcd_costs=0.5*(qcd_costs_j1+qcd_costs_j2)
        sig_costs=0.5*(sig_costs_j1+sig_costs_j2)
    else:
        print("Using both (j1,j2) loss")
        plot_dir=os.path.join(plot_dir,'none');pathlib.Path(plot_dir).mkdir(parents=True,exist_ok=True)
        qcd_fids=nnp.concatenate((qcd_fids_j1,qcd_fids_j2),axis=0)
        sig_fids=nnp.concatenate((sig_fids_j1,sig_fids_j2),axis=0)
        qcd_costs=nnp.concatenate((qcd_costs_j1,qcd_costs_j2),axis=0)
        sig_costs=nnp.concatenate((sig_costs_j1,sig_costs_j2),axis=0)

    qcd_labels=nnp.zeros(qcd_fids.shape[0])
    sig_labels=nnp.ones(sig_fids.shape[0])
    labels=nnp.concatenate([qcd_labels,sig_labels],axis=0)
    fids=nnp.concatenate([qcd_fids,sig_fids],axis=0)
    costs=nnp.concatenate([qcd_costs,sig_costs],axis=0) 
    try:
        effs=nnp.array([30,60,70,80,90,100])
        inv_effs=100-effs

        # Plot QCD shapes in various quantiles
        cuts=nnp.percentile(qcd_costs,effs)
        mjj=nnp.concatenate([qcd_features[:,0],sig_features[:,0]],axis=0)
        plt.hist(mjj[labels==0],bins=100,range=[1000,5000],label='QCD inclusive',histtype='step',linewidth=1.,density=True)
        for i,(lower_cut,upper_cut) in enumerate(zip(cuts[:-1],cuts[1:])):
            
            mask=(costs>=lower_cut)&(costs<upper_cut)
            cat_labels=labels[mask]
            cat_features=mjj[mask]
            if inv_effs[i+1]==100: leg=f'>Q{inv_effs[i]}'
            else: leg=f'Q{inv_effs[i]}-{inv_effs[i+1]}'
            plt.hist(cat_features[cat_labels==0],bins=100,range=[1000,5000],label=f'QCD: {leg}',histtype='step',linewidth=1.,density=True,alpha=0.8)
            #plt.hist(cat_features[cat_labels==1],bins=100,range=[1000,5000],label=ps.labels[cfg.signal],histtype='step',linewidth=2)
        plt.legend(loc='upper right')
        plt.minorticks_on()
        plt.grid(True,which='major',linestyle='--')
        plt.xlabel('$m_{JJ}$ (GeV)')
        plt.ylabel('No. of events')
        plt.title("Quantum VAE")
        plt.savefig(os.path.join(hist_dir,f'mjj_cut.png'))
        plt.clf()
        plt.scatter(mjj[labels==0],costs[labels==0],label='QCD',s=2,marker='o', alpha=.6,)
        plt.scatter(mjj[labels==1],costs[labels==1],label=ps.labels[cfg.signal],s=1,marker='s', alpha=.4,)
        #plt.hist(costs[labels==0],bins=100,range=[0,1],label='QCD',histtype='step',linewidth=2)
        #plt.hist(costs[labels==1],bins=100,range=[0,1],label=ps.labels[cfg.signal],histtype='step',linewidth=2)
        plt.legend(loc='upper left')
        plt.minorticks_on()
        plt.grid(True,which='major',linestyle='--')
        plt.xlabel('$m_{JJ}$ (GeV)')
        plt.ylabel('Cost = $1-$ \u27e8$T|R$\u27e9')
        plt.title('Signal: '+ps.labels[cfg.signal])
        plt.savefig(os.path.join(plot_dir,f'cost_vs_mJJ_{cfg.signal}.png'))
        plt.clf()
    #plt.ylabel('No. of events')
    #plt.xlabel('Cost = $1-$ \u27e8$T|R$\u27e9 (scaled to $[0,1]$)')
    # if inv_effs[i]==50.: 
    except:
        plt.clf()
        print("No mjj plots for jet-wise fidelity")


    #scaler = MinMaxScaler(feature_range=(0, 1.))
    #costs=scaler.fit_transform(costs.reshape(-1,1)).flatten()


    fpr,tpr,_=roc_curve(labels,costs)
    roc_auc=roc_auc_score(labels,costs)
    precision,recall,_ = precision_recall_curve(labels,costs)

    bins_qcd,edges_qcd=nnp.histogram(qcd_fids,density=True,bins=25,range=[90,100])
    bins_sig,edges_sig=nnp.histogram(sig_fids,density=True,bins=25,range=[90,100])
    plt.stairs(bins_qcd,edges_qcd,fill=True,label='QCD',alpha=0.6)
    plt.stairs(bins_sig,edges_sig,fill=False,label=ps.labels[cfg.signal])
    plt.minorticks_on()
    plt.grid(True,which='major',linestyle='--')

    plt.xlabel(u'Quantum Fidelity (%): \u27e8$T|R$\u27e9')
    plt.ylabel('No. of events')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plot_dir,f'fidelity_hist_{cfg.signal}.png'))

    plt.clf()
    plt.plot(fpr,tpr,label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {ps.labels[cfg.signal]}')
    plt.minorticks_on()
    plt.grid(True,which='major',linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(plot_dir,f'roc_curve_{cfg.signal}.png'))

    plt.clf()
    plt.plot(recall,precision,label='AUC = %0.3f' % roc_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall: {ps.labels[cfg.signal]}')
    plt.minorticks_on()
    plt.grid(True,which='major',linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(plot_dir,f'precision_recall_curve_{cfg.signal}.png'))
    
    # effs=nnp.arange(10,100,0.5)
    # sig_cuts=nnp.percentile(sig_costs,effs)
    # sig_cuts,idx=nnp.unique(sig_cuts,return_index=True)
    # effs=effs[idx]
    # significance=[]
    # plot_eff=[]
    sic=tpr/np.sqrt(fpr)
    
    # for cut in sig_cuts:
    #     sig_eff=nnp.sum(sig_costs>=cut)/sig_costs.shape[0]
    #     qcd_eff=nnp.sum(qcd_costs>=cut)/qcd_costs.shape[0]
    #     if sig_eff<0.15: continue
    #     if qcd_eff>0:
    #         plot_eff.append(sig_eff)
    #         significance.append(sig_eff/nnp.sqrt(qcd_eff))
    # sig_cuts=nnp.percentile(sig_costs,effs)
    
    #plt.plot(plot_eff,significance)
    plt.plot(tpr,sic)
    plt.minorticks_on()
    #plt.grid(True,which='major',linestyle='--')
    plt.xlabel('Signal efficiency')
    plt.ylabel('Significance Improvement')
    plt.xlim(0.1,1)
    plt.title('SIC: '+ps.labels[cfg.signal])
    plt.savefig(os.path.join(plot_dir,f'SIC_{cfg.signal}.png'))
    
    if log_wandb:
        for filename in glob.glob(os.path.join(plot_dir, "*.png")):
            wandb.log({os.path.split(filename)[-1].replace('*.png',''): wandb.Image(filename)})
    # Finish WandB run
        wandb.finish()
if __name__ == "__main__":
    main()
