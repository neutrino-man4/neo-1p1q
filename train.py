import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import subprocess
import datetime
import glob
import time
import matplotlib.pyplot as plt
import pennylane.numpy as np
import pennylane as qml
import helpers.utils as ut
import case_reader as cr
import quantum.losses as loss
from quantum.circuits.base import CircuitWeights
from loguru import logger
import wandb


@hydra.main(config_path="./hydra_configs/VQC", config_name="base")
def main(cfg: DictConfig):
    # Set up directories
    base_dir: str = cfg.base_dir
    save_dir = os.path.join(cfg.save_dir, cfg.seed)
    plot_dir = os.path.join(save_dir, 'plots')
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    try:
        run_str = f"{os.getlogin()}_{cfg.seed}"
    except:
        run_str = f"abal_{cfg.seed}"
    
    wandb.init(project="1P1Q", config=OmegaConf.to_container(cfg), name=run_str, notes=cfg.desc)
    with open(os.path.join(save_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb.run.id)

    # Logging setup
    logger.add(os.path.join(save_dir, 'logs.log'), rotation='10 MB', backtrace=True, diagnose=True, level='DEBUG', mode="w")
    logger.info("########################################### \n\n")
    logger.info(f"This circuit contains {cfg.wires} qubits")
    
    print("Will save models to: ", save_dir)

    # Save initial arguments for logging purposes
    ut.Pickle(cfg, 'args', path=save_dir)
    with open(os.path.join(save_dir, 'args.txt'), 'w+') as f:
        f.write(repr(cfg))

    # Further setup based on config
    if cfg.resume:
        test_args = ut.Unpickle(os.path.join(save_dir, 'args.pickle'))
        import importlib
        qc = importlib.import_module('saved_models.' + cfg.seed + '.FROZEN_ARCHITECTURE')
        model_path = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[-1]
        init_weights = ut.Unpickle(model_path)

        logger.add(os.path.join(cfg.save_dir, 'logs.log'), rotation='10 MB', backtrace=True, diagnose=True, level='DEBUG', mode="a")
        logger.info("########################################### \n\n")
        logger.info(f"Resuming training from last checkpoint at {model_path}")
        logger.info(f"Using arguments specified in original training run")
        logger.info(f"Training resumed at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        logger.info("Current weights are: ", init_weights)
        logger.info("\n\n ########################################### \n\n")
        cfg = test_args
        cfg.seed = str(cfg.seed)
    else:
        import quantum.architectures as qc
        logger.info("########################################### \n\n")
        logger.info(f"This circuit contains {cfg.wires} qubits")
        logger.info("\n\n ########################################### \n\n")
        tmpfile = os.path.join(save_dir, 'FROZEN_ARCHITECTURE.py')
        subprocess.run(['cp', os.path.join(base_dir, 'quantum/architectures.py'), tmpfile])
        tmpfile = os.path.join(save_dir, 'FROZEN_DATAREADER.py')
        subprocess.run(['cp', os.path.join(base_dir, 'case_reader.py'), tmpfile])
        tmpfile = os.path.join(save_dir, 'FROZEN_LOSS.py')
        subprocess.run(['cp', os.path.join(base_dir, 'quantum/losses.py'), tmpfile])
        # delete checkpoint directory contents
        try:
            subprocess.run(['rm', '-r', os.path.join(save_dir, 'checkpoints')])
        except:
            pass
        
    logger.info(f"Feature are scaled to the following limits: {ut.feature_limits}")

    if cfg.norm_pt:
        logger.info(f"pT will not be scaled to the above limit. Will be normalized using 1/jet_pt")
    else:
        logger.info(f"pT will also be scaled assuming above maxima")
    if cfg.flat:
        logger.info("Using flat mjj distribution for training")
    if cfg.loss == 'prob':
        cost_fn = loss.probabilistic_loss
    else:
        cost_fn = loss.VQC_cost
    if cfg.shots < 0:
        logger.warning("Negative shots specified. Setting to None for an analytic calculation.")
    # Create the quantum classifier instance
    VQC = qc.QuantumClassifier(
        wires=cfg.wires, 
        shots=cfg.shots if cfg.shots > 0 else None,
        dev_name=cfg.device_name,
        layers=cfg.num_layers,
        backend_name=cfg.backend,
        test=False  # Don't set circuit immediately
    )
    VQC.set_circuit(circuit_type=cfg.circuit_type, operations_per_qubit=cfg.operations_per_qubit)

    if not cfg.resume:
        shape = VQC._impl.rotation_shape(len(VQC.auto_wires), cfg.num_layers)
        rot = np.array(np.random.uniform(0, np.pi, size=(shape.L, shape.N, shape.R)), requires_grad=True)
        aux = {**VQC._impl.aux_defaults, **dict(cfg.get('aux_weights', {}))}
        aux = {k: np.array(v, requires_grad=True) for k, v in aux.items()}
        init_weights = CircuitWeights(rot=rot, aux=aux)

    train_max_n = cfg.n_signal + cfg.n_background
    valid_max_n = cfg.n_signal_val + cfg.n_background_val
    
    # Print training parameters using the VQC instance method
    VQC.print_training_params()

    # Save initial arguments for logging purposes
    ut.Pickle(cfg, 'args', path=save_dir)
    with open(os.path.join(save_dir, 'args.txt'), 'w+') as f:
        f.write(repr(cfg))

    # Load the data: balanced signal vs background, per the config keys.
    # JetClass layout is <data_dir>/<split>/<sample>/<sample>_NNN.h5.
    train_split = 'flat_train' if cfg.flat else 'train'
    val_split = 'flat_val' if cfg.flat else 'val'
    _class_files = lambda split, sample: sorted(glob.glob(os.path.join(cfg.data_dir, split, sample, '*.h5')))
    train_sig, train_bg = _class_files(train_split, cfg.signal), _class_files(train_split, cfg.background)
    val_sig, val_bg = _class_files(val_split, cfg.signal), _class_files(val_split, cfg.background)
    num_particles = getattr(cfg, 'num_particles', len(VQC.auto_wires))
    logger.info(f"Number of particles to load: {num_particles}")

    if not (train_sig and train_bg and val_sig and val_bg):
        raise FileNotFoundError(
            f"Missing JetClass files under {cfg.data_dir} for signal='{cfg.signal}', "
            f"background='{cfg.background}' (splits '{train_split}', '{val_split}')"
        )
    logger.info(f"Training set: {cfg.n_signal} '{cfg.signal}' + {cfg.n_background} '{cfg.background}' jets from {train_split}/")
    logger.info(f"Validation set: {cfg.n_signal_val} '{cfg.signal}' + {cfg.n_background_val} '{cfg.background}' jets from {val_split}/")

    train_loader = cr.OneP1QDataLoader(
        signal_filelist=train_sig, background_filelist=train_bg,
        n_signal=cfg.n_signal, n_background=cfg.n_background,
        batch_size=cfg.batch_size,
        input_shape=(num_particles, 3),
        train=True,
        normalize_pt=cfg.norm_pt,
        logger=logger,
    )
    val_loader = cr.OneP1QDataLoader(
        signal_filelist=val_sig, background_filelist=val_bg,
        n_signal=cfg.n_signal_val, n_background=cfg.n_background_val,
        batch_size=cfg.batch_size,
        input_shape=(num_particles, 3),
        train=False,
        normalize_pt=cfg.norm_pt,
        logger=logger,
    )

    # Initialize the optimizer
    optimizer = qml.AdamOptimizer(stepsize=cfg.lr)

    # Initialize the trainer with the new class signature
    trainer = qc.QuantumTrainer(
        model=VQC,
        lr=cfg.lr,
        optimizer=optimizer,
        loss_fn=cost_fn,
        save=cfg.save,
        train_max_n=train_max_n,
        valid_max_n=valid_max_n,
        epochs=cfg.epochs,
        patience=cfg.patience,
        improv=cfg.improv,
        wandb=wandb,
        lr_decay=cfg.lr_decay,
        loss_type=cfg.loss,
        # Pass additional parameters via kwargs
        init_weights=init_weights,
        batch_size=cfg.batch_size,
        logger=logger
    )

    trainer.print_params('Initialized parameters!')
    trainer.set_directories(save_dir)

    if cfg.resume:
        trainer.set_current_epoch(ut.get_current_epoch(model_path))
        logger.info(f"Resuming training from epoch {trainer.current_epoch}")
    else:
        logger.info(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        logger.info(f'Epochs: {cfg.epochs} | Learning rate: {cfg.lr} | Batch size: {cfg.batch_size} \nBackend: {cfg.backend} | Wires: {cfg.wires} | Shots: {cfg.shots} \n')    
        logger.info(f'Additional information: {cfg.desc}')

    if cfg.evictable:
        trainer.is_evictable_job(seed=cfg.seed)

    # Begin training
    abs_start = time.time()
    try:
        history = trainer.run_training_loop(train_loader, val_loader)
    except KeyboardInterrupt:
        print("WHYYYYY")
        print("DON'T PRESS CTRL+C AGAIN. I'M TRYING TO SAVE THE CURRENT MODEL AND WRITE TO LOG!")
        trainer.save(save_dir, name='aborted_weights.pickle')
        trainer.print_params('Training aborted. Current parameters are: ')
    finally:
        logger.info('Training completed with the following parameters:')
        trainer.print_params('Trained parameters:')
        history = trainer.fetch_history()
        print(history)
        if cfg.save:
            done_epochs = len(history['train'])
            ut.Pickle(history, 'history', path=save_dir)
            fig, axes = plt.subplots(figsize=(15, 12))
            axes.plot(np.arange(done_epochs), history['train'], label='train', linewidth=2)
            axes.plot(np.arange(done_epochs + 1), history['val'], label='val', linewidth=2)
            axes.set_xlabel('Epochs', size=25)
            axes.set_ylabel('$1-<T|F> $(in %)', size=25)
            axes.set_xticks(np.arange(0, done_epochs + 1, 5))
            axes.legend(prop={'size': 25})

            axes.tick_params(labelsize=20)
            fig.savefig(os.path.join(save_dir, 'history'))
            abs_end = time.time()
            logger.info(f"Training finished at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
            logger.info(f"Total time taken including all overheads: {abs_end - abs_start:.2f} seconds")

        # Close WandB run
        wandb.finish()


if __name__ == "__main__":
    main()