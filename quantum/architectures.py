# pylint: disable=maybe-no-member
from typing import Optional, Callable, Union, List, Dict, Tuple, Any
import pennylane as qml
from helpers.utils import getIndex
from quantum.circuits.base import Circuit, CircuitWeights
from quantum.circuits import registry
from itertools import combinations
import time
from tqdm import tqdm
import pennylane.numpy as np
import os
import pathlib
import helpers.utils as ut
import subprocess
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import json


class QuantumClassifier:
    """
    A Quantum Classifier that uses quantum circuits for classification tasks.
    
    This class encapsulates all the functionality needed to create, configure,
    and run quantum classification circuits using PennyLane.
    
    Args:
        wires: Number of qubits to use in the circuit
        layers: Number of layers in the quantum circuit
        shots: Number of shots for measurements on quantum states
        dev_name: Name of the quantum device to use
        use_ancilla: Whether to use an ancilla qubit
        backend_name: Backend for the QNode (e.g., 'autograd', 'torch', 'jax')
        test: If True, sets the circuit immediately for testing
    """
    
    def __init__(
        self, 
        wires: int = 4,
        layers: int = 1,
        shots: int = 5000,
        dev_name: str = 'default.qubit',
        use_ancilla: bool = False,
        backend_name: str = 'autograd',
        test: bool = False,
        **kwargs: Any
    ) -> None:
        # Circuit configuration
        self.n_qubits = wires
        self.num_layers = layers
        self.use_ancilla = use_ancilla
        self.backend = backend_name
        self.n_trash_qubits = -1
        self.separate_ancilla = False
        # Initialize wire configurations
        self._initialize_wires()
        
        # Set up device
        self.device = self._set_device(shots=shots, device_name=dev_name)
        
        # Circuit and weights
        self._impl: Optional[Circuit] = None
        self.circuit: Optional[qml.QNode] = None
        self.current_weights: Optional[CircuitWeights] = None
        
        # Data indices for particle properties
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'), 
            'pt': getIndex('particle', 'pt')
        }
        
        if test:
            self.set_circuit()
            try:
                self.total_batches=kwargs['read_n']
                print(f"Will read {self.total_batches} batches with batch size 1")
            except:
                self.total_batches=100000
                print("WARNING: total_batches not set, using default value of 100000")
    
    def _initialize_wires(self) -> None:
        """
        Initialize wire (qubit) indices and create necessary combinations.
        """
        total_qubits = self.n_qubits

        if self.use_ancilla:
            total_qubits += 1
            self.truth_wire = total_qubits - 1  # Last wire is the truth wire
        else:
            self.truth_wire = None

        self.all_wires = list(range(total_qubits))
        self.auto_wires = list(range(self.n_qubits))
        self.two_comb_wires = list(combinations(range(self.n_qubits), 2))
    
    def _set_device(self, shots: int, device_name: str) -> "qml.devices.Device":
        """
        Set up the quantum device for simulation/execution.
        
        Args:
            shots: Number of shots for each measurement
            device_name: Name of the quantum device
            
        Returns:
            Initialized PennyLane device
        """
        device = qml.device(device_name, wires=len(self.all_wires), shots=shots)
        print(f"Device initialized: {device}")
        return device
    
    def print_training_params(self) -> None:
        """
        Print initialized training parameters for verification.
        Includes a pause to allow user review.
        """
        print("\n Sanity check: \n")
        print('all_wires:', self.all_wires)
        print('auto_wires:', self.auto_wires)  
        print('two_comb_wires:', self.two_comb_wires)
        print('no. of layers:', self.num_layers)
        print('index:', self.index)
        print('Using truth wire (if None, then unused):', self.truth_wire)
        print('\n' + '#' * 46 + '\n')
        print("Sleep on it for 3s")
        print("Maybe you want to change something?")
        print("Then press CTRL-C")
        print('\n' + '#' * 50 + '\n')
        time.sleep(3)
        print("LETS GOOOOOOOOOOOOO")
        time.sleep(1)
    
    def set_circuit(self, circuit_type: str = 'normal', operations_per_qubit: Optional[int] = None) -> None:
        """
        Configure the QNode circuit from the circuit registry.

        Args:
            circuit_type: registered circuit name (see quantum.circuits.registry).
            operations_per_qubit: overrides the circuit's default rotation ops
                per qubit per layer (R in its RotationShape), if given.
        """
        self._impl = registry.get(circuit_type, self.num_layers)
        if operations_per_qubit is not None:
            self._impl.operations_per_qubit = operations_per_qubit
        self.circuit = qml.QNode(
            lambda weights, inputs: self._impl.build(weights, inputs, self.auto_wires),
            self.device,
            interface=self.backend
        )

    def fetch_circuit(self) -> qml.QNode:
        """
        Get the quantum circuit for inference or training.

        Returns:
            Configured quantum node (QNode) circuit
        """
        if self.circuit is None:
            self.set_circuit()
        return self.circuit

    def fetch_backend(self) -> str:
        """
        Get the backend being used for the QNode.
        
        Returns:
            Name of the backend
        """
        return self.backend
    
    def load_weights(self, model_path: str, train: bool = False,rearrange=False) -> None:
        """
        Load pre-trained weights for the classifier.
        
        Args:
            model_path: Path to the file containing the pre-trained model
            train: If True, enables gradients for the weights
        """
        dictionary = ut.Unpickle(model_path)
        saved_weights = dictionary['weights']
        if isinstance(saved_weights, CircuitWeights):
            # D7: split-weight checkpoints carry their own rot/aux structure --
            # just re-enable gradients on each trainable piece, nothing to rearrange.
            self.current_weights = CircuitWeights(
                rot=np.array(saved_weights.rot, requires_grad=train),
                aux={k: np.array(v, requires_grad=train) for k, v in saved_weights.aux.items()},
            )
            return
        self.current_weights = np.array(saved_weights, requires_grad=train)
        if rearrange:
            x_weights= self.current_weights[:self.n_qubits*self.num_layers]
            y_weights = self.current_weights[self.n_qubits*self.num_layers:2*self.n_qubits*self.num_layers]
            z_weights = self.current_weights[2*self.n_qubits*self.num_layers:3*self.n_qubits*self.num_layers]
            self.current_weights = np.zeros_like(self.current_weights)

            self.current_weights[2:3*self.n_qubits*self.num_layers:3] = x_weights
            self.current_weights[:3*self.n_qubits*self.num_layers:3] = z_weights
            self.current_weights[1:3*self.n_qubits*self.num_layers:3] = y_weights
    def print_weights(self) -> None:
        """Print the current weights of the quantum classifier."""
        print('Current weights: \n\n', self.current_weights)
    
    def run_inference(
        self, 
        dataloader: DataLoader, 
        loss_fn: Callable,
        loss_type: str = 'BCE'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on the classifier circuit using loaded weights and a dataloader.
        
        Args:
            dataloader: DataLoader containing input data and labels (batch_size=1)
            loss_fn: Loss function to calculate the quantum cost (should be VQC_cost)
            loss_type: Type of loss function to use
            
        Returns:
            Tuple of (costs, scores) both with shape (N_inputs,)
            
        Raises:
            ValueError: If weights are not initialized
        """
        if self.current_weights is None:
            raise ValueError(
                'Weights not initialized. Load a model first by calling load_weights(model_path)'
            )
        
        all_costs = []
        all_scores = []
        all_labels = []
        for batch_inputs, batch_labels in tqdm(dataloader, desc="Running inference",total=self.total_batches):
            # batch_inputs.shape = (1, n_qubits, 3)
            # batch_labels.shape = (1,)
            
            # Compute cost and score for this single sample
            cost, score = loss_fn(
                self.current_weights,
                inputs=batch_inputs,
                labels=batch_labels,
                quantum_circuit=self.circuit,
                return_scores=True,
                loss_type=loss_type
            )
            
            all_costs.append(float(cost))
            all_scores.append(float(score))
            all_labels.append(float(batch_labels[0]))
        # Convert to numpy arrays
        costs = np.array(all_costs)    # Shape: (N_inputs,)
        scores = np.array(all_scores)  # Shape: (N_inputs,)
        labels = np.array(all_labels)  # Shape: (N_inputs,)
        
        print("Inference completed")
        return costs, scores, labels
class QuantumTrainer:
    """
    A class for training quantum classification circuits.
    
    Args:
        model: The VQC circuit to be trained
        lr: Learning rate for the optimizer
        optimizer: The optimizer used for training
        loss_fn: The loss function used for optimization
        save: Whether to save the trained model and checkpoints
        train_max_n: Maximum number of training samples
        valid_max_n: Maximum number of validation samples  
        epochs: Number of training epochs
        patience: Patience for early stopping
        improv: Minimum improvement threshold for early stopping
        wandb: Weights & Biases logger instance
        lr_decay: Whether to apply learning rate decay
        loss_type: Type of loss function ('BCE', etc.)
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        model: QuantumClassifier,
        lr: float = 0.001,
        optimizer: Optional[Callable] = None,
        loss_fn: Optional[Callable] = None,
        save: bool = True,
        train_max_n: int = 10000,
        valid_max_n: int = 2000,
        epochs: int = 20,
        patience: int = 2,
        improv: float = 0.01,
        wandb: Optional[Any] = None,
        lr_decay: bool = False,
        loss_type: str = 'MSE',
        **kwargs: Any
    ) -> None:
        self.model = model
        self.circuit = model.fetch_circuit()
        self.backend = model.fetch_backend()
        
        # Training parameters
        self.init_weights = kwargs.get('init_weights')
        self.batch_size = kwargs.get('batch_size', 1000)
        self.logger = kwargs.get('logger')
        
        # Training configuration
        self.train_max_n = train_max_n
        self.valid_max_n = valid_max_n
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.patience = patience
        self.saving = save
        self.loss_type = loss_type
        self.improv = improv
        self.is_evictable = False
        self.wandb = wandb
        
        # Training state
        self.current_weights = self.init_weights
        self.current_epoch = 0
        self.optim = optimizer
        self.quantum_loss = loss_fn
        self.history: Dict[str, List[float]] = {'train': [], 'val': [], 'auc': []}
        
        # Directories (to be set later)
        self.save_dir: Optional[str] = None
        self.checkpoint_dir: Optional[str] = None
        self.seed: Optional[Any] = None
        
        print(f'Performing optimization with: {self.optim} | Setting Learning rate: {lr}')
        print('Backend:', self.backend, '\n')
    
    def iteration(
        self, 
        data: np.ndarray, 
        labels: np.ndarray, 
        train: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Perform a single training or validation iteration.
        
        Args:
            data: Batch of input data
            labels: Corresponding labels
            train: Whether to perform training (True) or validation (False)
            
        Returns:
            Training loss (if train=True) or tuple of (validation loss, scores)
        """
        if train:
            # step_and_cost needs each trainable piece as its own argument (a
            # CircuitWeights isn't itself differentiable) -- see D7. cost_fn
            # repacks rot + aux back into a CircuitWeights for the real loss.
            aux_keys = list(self.current_weights.aux.keys())

            def cost_fn(rot: np.ndarray, *aux_values: np.ndarray) -> np.ndarray:
                weights = CircuitWeights(rot=rot, aux=dict(zip(aux_keys, aux_values)))
                return self.quantum_loss(
                    weights,
                    inputs=data,
                    labels=labels,
                    quantum_circuit=self.circuit,
                    loss_type=self.loss_type
                )

            updated_args, cost = self.optim.step_and_cost(
                cost_fn,
                self.current_weights.rot,
                *(self.current_weights.aux[k] for k in aux_keys)
            )
            new_rot, *new_aux_values = updated_args
            self.current_weights = CircuitWeights(rot=new_rot, aux=dict(zip(aux_keys, new_aux_values)))
            return float(cost)
        else:
            cost, scores = self.quantum_loss(
                self.current_weights,
                inputs=data,
                labels=labels,
                quantum_circuit=self.circuit,
                return_scores=True,
                loss_type=self.loss_type
            )
            return float(cost), float(scores)
    
    def is_evictable_job(self, seed: Optional[Any] = None) -> None:
        """
        Mark the current job as evictable and enable checkpoint copying to EOS.
        
        Args:
            seed: Random seed for checkpoint saving
        """
        self.is_evictable = True
        self.seed = seed
    
    def run_training_loop(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Execute the full training loop, including training and validation.
        
        Args:
            train_loader: DataLoader for the training dataset
            val_loader: DataLoader for the validation dataset
            
        Returns:
            Training and validation history (losses and accuracies)
        """
        self.print_params('Initial weights: ')
        n_decays = 0
        last_decay = 0
        complete = False
        
        for n_epoch in tqdm(range(self.epochs + 1)):
            sample_counter = 0
            batch_yield = 0
            self.current_epoch = n_epoch
            losses = 0.0
            
            # Early stopping logic
            if n_epoch > 4:
                recent_val_metrics = self.history['auc'][-2:]
                previous_val_metric = self.history['auc'][-3]
                improvement = np.mean(recent_val_metrics) - previous_val_metric
                
                if improvement < self.improv:
                    if self.lr_decay:
                        if (n_decays < self.patience) and ((n_epoch - last_decay) >= 2):
                            last_decay = self.current_epoch
                            n_decays += 1
                            self.optim.stepsize *= 0.5
                            self.logger.info(
                                f'No improvement observed over last 3 epochs. \n'
                                f'Learning rate decayed to {self.optim.stepsize} at epoch {n_epoch}'
                            )
                        elif n_decays >= self.patience:
                            self.logger.info(
                                f"\n\nNo improvement over last 3 epochs and {self.patience} "
                                f"decay steps. Early stopping!\n\n"
                            )
                            self.save(self.save_dir, name='trained_model.pickle')
                            complete = True
                            break
                    else:
                        self.logger.info(
                            "\n\nNo improvement over last 3 epochs. Early stopping!\n\n"
                        )
                        self.save(self.save_dir, name='trained_model.pickle')
                        complete = True
                        break
            
            # Training phase
            if n_epoch > 0:
                print("Start Training")
                start = round(time.time(), 2)
                
                for data, labels in tqdm(
                    train_loader, 
                    total=int(self.train_max_n / self.batch_size)
                ):
                    sample_counter += data.shape[0]
                    batch_yield += 1
                    loss = self.iteration(data, labels=labels, train=True)
                    losses += loss
                    
                    if self.wandb is not None:
                        self.wandb.log({'train_loss': losses / batch_yield})
                
                end = round(time.time(), 2)
                train_loss = losses / batch_yield
                self.print_params('Current weights: \n\n')
                print('Now validating!')
            else:
                print('Running initial validation pass')
            
            # Validation phase
            val_loss = 0.0
            val_batch_yield = 0
            val_score = []
            val_labels = []
            
            for data, labels in tqdm(
                val_loader, 
                total=int(self.valid_max_n / self.batch_size)
            ):
                loss, score = self.iteration(data, labels=labels, train=False)
                val_loss += loss
                val_score.append(score)
                val_labels.append(labels)
                val_batch_yield += 1
            
            val_loss = val_loss / val_batch_yield
            val_labels = np.array(val_labels).flatten()
            val_score = np.array(val_score).flatten()
            val_auc = roc_auc_score(val_labels, val_score)
            val_std = np.std(val_score)
            val_score_mean = np.mean(val_score)
            
            if self.wandb is not None:
                self.wandb.log({'val_loss': val_loss, 'val_auc': val_auc})
            
            # Logging
            if n_epoch > 0:
                self.logger.info(
                    f'Epoch {n_epoch}: Network with {len(self.model.auto_wires)} input qubits '
                    f'trained on {sample_counter} samples in {batch_yield} batches'
                )
                self.logger.info(
                    f'Epoch {n_epoch}: Train Loss = {train_loss:.3f} | Val loss = {val_loss:.3f} | '
                    f'Val preds mean and std = {val_score_mean:.3f}, {val_std:.3f} | '
                    f'Val AUC = {val_auc:.3f} | Time taken = {end-start:.3f} seconds\n\n'
                )
                self.history['train'].append(train_loss)
            else:
                self.logger.info('Initial validation pass completed')
                self.logger.info(
                    f'Epoch {n_epoch} (No training performed): Val loss = {val_loss:.3f} | '
                    f'Val AUC = {val_auc:.3f} | Val preds mean and std = {val_score_mean:.3f}, {val_std:.3f}\n\n'
                )
            
            self.history['val'].append(val_loss)
            self.history['auc'].append(val_auc)
            
            # Saving
            if self.saving:
                name = None
                if n_epoch == self.epochs:
                    name = 'trained_model.pickle'
                elif n_epoch == 0:
                    name = 'init_weights.pickle'
                
                self.save(self.save_dir, name=name)
                
                if self.is_evictable and n_epoch > 0:
                    print('Will copy over checkpoints')
                    checkpoint_name = f'ep{self.current_epoch:02}.pickle'
                    try:
                        tmpfile = (
                            f"{os.environ['EOS_MGM_URL']}://eos/user/"
                            f"{os.environ['CERN_USERNAME'][0]}/{os.environ['CERN_USERNAME']}/"
                            f"QML/checkpoint_dumps/{self.seed}/{checkpoint_name}"
                        )
                        exec_path = os.path.join(os.environ['BELLE2_EXEC'], 'xrdcp')
                        subprocess.call(
                            f'{exec_path} {os.path.join(self.checkpoint_dir, checkpoint_name)} {tmpfile}',
                            shell=True
                        )
                    except Exception:
                        print("Failed to copy over checkpoints")
        
        if not complete:
            ut.Pickle(self.history, 'history.pickle', path=self.save_dir)
        
        return self.history
    
    def print_params(self, prefix: Optional[str] = None) -> None:
        """
        Print the current parameters (weights) of the quantum classifier.
        
        Args:
            prefix: Optional prefix to print before the parameters
        """
        if prefix is not None:
            print(prefix)
        print('autograd weights:', self.current_weights, '\n')
    
    def save(self, save_dir: str, name: Optional[str] = None) -> None:
        """
        Save the model weights to a specified directory.
        
        Args:
            save_dir: Directory where the model weights will be saved
            name: The name of the file to save. If not provided, 
                  the file name will be based on the current epoch
        """
        opt_name = None
        
        if name is None:
            if self.current_epoch > 100:
                name = f'ep{self.current_epoch:03}.pickle'
                opt_name = f'optimizer_ep{self.current_epoch:03}.json'
            else:
                name = f'ep{self.current_epoch:02}.pickle'
                opt_name = f'optimizer_ep{self.current_epoch:02}.json'
        
        if 'trained' not in name:
            save_dir = self.checkpoint_dir
        else:
            opt_name = 'optimizer.json'
        
        ut.Pickle({'weights': self.current_weights}, name, path=save_dir)
        
        # Save optimizer state
        try:
            optim_dict = {
                'stepsize': self.optim.stepsize,
                'beta1': self.optim.beta1,
                'beta2': self.optim.beta2,
                'epsilon': self.optim.eps,
                'fm': self.optim.fm,
                'sm': self.optim.sm,
                't': self.optim.t
            }
            
            with open(os.path.join(save_dir, opt_name), 'w') as f:
                json.dump(optim_dict, f)
            print("Optimizer state saved")
        except Exception as e:
            print(f"Error saving optimizer state: {str(e)}")
    
    def get_current_epoch(self) -> int:
        """
        Get the current epoch number during training.
        
        Returns:
            The current epoch number
        """
        return self.current_epoch
    
    def set_current_epoch(self, epoch: int) -> None:
        """
        Set the current epoch number if training is resumed from a checkpoint.
        
        Args:
            epoch: The epoch number to set
        """
        self.current_epoch = epoch + 1
        print("Resume training from epoch:", epoch + 1)
    
    def set_directories(self, save_dir: str) -> None:
        """
        Set up directories for saving model checkpoints and logs.
        
        Args:
            save_dir: The directory where model and checkpoints will be saved
        """
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        pathlib.Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def fetch_history(self) -> Dict[str, List[float]]:
        """
        Get the history of training and validation losses and accuracies.
        
        Returns:
            Dictionary containing the history of training and validation metrics
        """
        return self.history