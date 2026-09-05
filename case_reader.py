'''
Date: August 2024
Author: Aritra Bal, ETP
Description: This script contains the data loader classes for the MC Datasets used in the studies performed for the CMS analysis EXO-22-026
'''

import h5py
import helpers.utils as ut
from pennylane import numpy as np
from sklearn.preprocessing import MinMaxScaler
import __main__
import numpy as nnp
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Tuple, Union
import loguru
eta_lims=ut.feature_limits['eta']
phi_lims=ut.feature_limits['phi']
class CASEJetClassDataset(IterableDataset):
    """
    Iterable dataset that streams single-jet data from JetClass-style .h5 files.

    Each file holds `jetConstituentsList` of shape (N, 100, 3) -- N jets, up to
    100 particles each, (eta, phi, pt) per particle -- and `jetFeatures` of
    shape (N, 10). Truth labels come from `truth_labels` if present, otherwise
    they are inferred from the filename (a name containing "zjetsto" is treated
    as QCD-like / label 0, anything else as label 1).

    Args:
        filelist (List[str]): List of file paths to the .h5 files.
        batch_size (int): Number of samples in each batch.
        max_samples (int): Maximum number of samples to load.
        data_key (str): Key to access the per-particle data inside the .h5 files.
        feature_key (str): Key to access jet-level features.
        input_shape (tuple[int]): (number of particles to read per jet, features per particle).
        epsilon (float): Small constant to avoid division by zero.
        train (bool): Whether to yield only training data, or labels as well.
        selection (str): Constituent-selection mode passed through to helpers.

    Yields:
        Tuple[np.ndarray, np.ndarray]: a batch of data and its integer labels.
    """
    def __init__(self, filelist:List[str]=None, batch_size:int=32, max_samples:int=5e4, data_key='jetConstituentsList',\
                 feature_key='jetFeatures',input_shape:tuple[int]=(10, 3),epsilon:float=1.0e-4,train:bool=True,\
                    normalize_pt:bool=False,logger=None,selection='equal'):
        super().__init__()
        self.filelist = sorted(filelist)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pt_index=ut.getIndex('particle','pt')
        self.eta_index=ut.getIndex('particle','eta')
        self.phi_index=ut.getIndex('particle','phi')
        self.jpt_index=ut.getIndex('jet','jet_pt')
        self.j_msd_index=ut.getIndex('jet','jet_sdmass')
        self.selection=selection
        self.max_samples = max_samples
        self.epsilon=epsilon
        self.data_key=data_key
        self.feature_key=feature_key
        self.train=train
        self.batch_counter=0
        self.normalize_pt=normalize_pt
        self.logger=logger
        self.n_qubits=input_shape[0]
        self.rng = np.random.default_rng(seed=None)
        # fresh entropy is pulled --> but only once for each instance of the dataloader.
    def fixed_rescale(self,data: np.ndarray, epsilon: float = 1.0e-4, type='pt') -> np.ndarray:
        """
        Rescales the data to a specified range. Instead of using the min/max values of the data array,
        uses fixed values instead, which can be modified in the assumed_limits array

        Args:
            data (np.ndarray): Input data to rescale.
            epsilon (float): Small offset to prevent numerical instability.

        Returns:
            np.ndarray: Rescaled data.
        """
        min=ut.feature_limits[type]['min']
        max=ut.feature_limits[type]['max']
        max-=epsilon
        assumed_limits=ut.assumed_limits
        if type not in ['pt','eta','phi']:
            raise NameError("Type must be either of [pt,eta,phi]")
        if self.logger is not None:
            self.logger.info(f"Assuming fixed sample maxima: [{assumed_limits[type][0]},{assumed_limits[type][1]}] for variable {type}")
        else:
            print(f"Assuming fixed sample maxima: [{assumed_limits[type][0]},{assumed_limits[type][1]}] for variable {type}")
        data_shape = data.shape
        data_reshaped = data.flatten()
        data_scaled = ((data_reshaped - assumed_limits[type][0])/(assumed_limits[type][1]-assumed_limits[type][0]))*(max-min) + min # scale using fixed values of 
        return data_scaled.reshape(data_shape[0], data_shape[1])
    
    def load_and_preprocess_file(self, file_path:str,inference:bool=False):
        """
        Loads and preprocesses a single JetClass-style .h5 file.

        Args:
            file_path (str): Path to the .h5 file.
            inference (bool): Whether the data is being loaded for inference.

        Returns:
            np.ndarray: Per-jet particle data, shape (N, n_qubits, 3).
            np.ndarray: Integer truth labels, shape (N,).
        """
        with h5py.File(file_path, 'r') as file:
            # jetConstituentsList has shape (N, 100, 3): N jets, up to 100 particles, (eta, phi, pt).
            jet_features=np.array(file[self.feature_key])
            jet_pt=jet_features[:,self.jpt_index]

            if self.logger is not None:
                self.logger.info(f"Reading hardest {self.n_qubits} PFCands per jet")
            else:
                print(f"Reading hardest {self.n_qubits} PFCands per jet")
            jet_etaphipt = np.array(file[self.data_key][()])
            sorted_indices = np.argsort(-jet_etaphipt[...,self.pt_index], axis=-1)
            jet_etaphipt = np.take_along_axis(jet_etaphipt, sorted_indices[...,None], axis=1)
            jet_etaphipt = jet_etaphipt[:,:self.n_qubits,:]

            try:
                truth_label = np.array(file['truth_labels'][()])
            except KeyError:
                if 'zjetsto'.casefold() in file_path.casefold():
                    truth_label = np.zeros(jet_etaphipt.shape[0])
                    print("Inferred: QCD like jets")
                else:
                    truth_label = np.ones(jet_etaphipt.shape[0])
        truth_label=np.array(truth_label,dtype=np.integer)

        if self.normalize_pt:
            print("Normalizing PFCand pT by jet pT")
            jet_etaphipt[...,self.pt_index]=jet_etaphipt[...,self.pt_index]/jet_pt[:,np.newaxis]
        else:
            jet_etaphipt[...,self.pt_index]=self.fixed_rescale(jet_etaphipt[...,self.pt_index], epsilon=self.epsilon,type='pt')

        jet_etaphipt[...,self.eta_index]=self.fixed_rescale(jet_etaphipt[...,self.eta_index], epsilon=self.epsilon,type='eta')
        jet_etaphipt[...,self.phi_index]=self.fixed_rescale(jet_etaphipt[...,self.phi_index], epsilon=self.epsilon,type='phi')

        print("sample max pt: ",np.max(jet_etaphipt[:,:,self.pt_index]))
        print("sample min pt: ",np.min(jet_etaphipt[:,:,self.pt_index]))
        print("sample max eta: ",np.max(jet_etaphipt[:,:,self.eta_index]))
        print("sample min eta: ",np.min(jet_etaphipt[:,:,self.eta_index]))
        print("sample max phi: ",np.max(jet_etaphipt[:,:,self.phi_index]))
        print("sample min phi: ",np.min(jet_etaphipt[:,:,self.phi_index]))
        if inference:
            return jet_etaphipt,jet_features,truth_label
        return jet_etaphipt, truth_label

    def load_for_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads data from the specified .h5 files for inference.

        Reads files until `max_samples` jets have been collected.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - jet_array (np.ndarray): per-jet particle data, shape (N, n_qubits, 3).
                - jetFeatures_array (np.ndarray): jet-level features, shape (N, 10).
                - truth_labels (np.ndarray): truth labels, shape (N,).
        """
        print(f"Will read a total of {self.max_samples} events for inference")
        jetFeatures_array=[]
        while len(jetFeatures_array)<self.max_samples:
            for i,file_path in enumerate(self.filelist):
                jet_etaphipt,jet_features,truth_label= self.load_and_preprocess_file(file_path,inference=True)
                if i==0:
                    jetFeatures_array=jet_features
                    jet_array=jet_etaphipt
                    truth_labels=truth_label
                else:
                    jetFeatures_array=np.concatenate([jetFeatures_array,jet_features],axis=0)
                    jet_array=np.concatenate([jet_array,jet_etaphipt],axis=0)
                    truth_labels=np.concatenate([truth_labels,truth_label],axis=0)
                if len(jetFeatures_array)>=self.max_samples:
                    break
        jet_array=jet_array[:self.max_samples]
        jetFeatures_array=jetFeatures_array[:self.max_samples]
        truth_labels=truth_labels[:self.max_samples]
        return np.array(jet_array,requires_grad=False),np.array(jetFeatures_array,requires_grad=False),np.array(truth_labels,requires_grad=False)

    def __iter__(self)-> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        '''
        Iterator that yields batches of data
        '''
        sample_counter=0
        #self.batch_counter=0
        for file_path in self.filelist:
            data, labels = self.load_and_preprocess_file(file_path)
            # Shuffle the data from this file only if training on a per-jet basis, otherwise it becomes necessary to preserve the order of jets
            
            #indices = np.arange(data.shape[0])
            #self.rng.shuffle(indices)
            #print("Loaded data was shuffled")
            #data = data[indices]
            #labels = labels[indices]
            
            # Yield data in batches
            for i in range(0, len(data), self.batch_size):
                if sample_counter >= self.max_samples:
                    return
                
                end = i + self.batch_size
                if end > len(data):
                    end = len(data)
                
                batch_data = data[i:end]
                batch_labels = labels[i:end]
                
                batch_data = torch.from_numpy(batch_data).float()
                #batch_labels = torch.from_numpy(batch_labels).float() # Assuming labels are floats
                
                sample_counter += batch_data.shape[0]
                yield np.array(batch_data,requires_grad=False), np.array(batch_labels,dtype=np.integer,requires_grad=False)

def OneP1QDataLoader(input_shape:tuple[int]=(100, 3),train:bool=True,**kwargs) -> DataLoader:
    '''
    Wrapper function to create a DataLoader for the CASEJetClassDataset.
    Args:
        filelist (List[str]): List of file paths to the .h5 files containing the data.
        batch_size (int): Number of samples in each batch.
        input_shape (tuple[int]): Shape of the input data, with the first element being the no. of particles per jet to read and the 2nd typically being (eta,phi,pt).
    Returns:
        DataLoader: Torch DataLoader object that yields batches of data.
    '''
    print(f"Will read only {input_shape[0]} particles per jet")
    dset = CASEJetClassDataset(input_shape=input_shape,train=train,**kwargs)
    return DataLoader(dset, batch_size=None)  # None for batch_size since batching is managed by the dataset


def select_subjet_constituents(jet_etaphipt, num_PFCands_subleading_jet, evt_subjet_idx, n_qubits=8,selection='equal'):
    """
    Selects a fixed number of particles per jet, filling with random sampling if needed.

    Parameters:
    - num_PFCands_subleading_jet: (N,) array representing number of particles in the lower pT subjet.
    - evt_subjet_idx: (N, 100) array containing indices of particles assigned to subjets.
    - jet_etaphipt: (N, 100, 3) array representing momenta (eta, phi, pT) of particles.
    - n_qubits: Desired number of particles per jet (default is 8).

    Returns:
    - jet_etaphipt_selected: (N, n_qubits, 3) array of selected particles per jet.
    """
    if n_qubits%2!=0:
        raise ValueError("No. of particles to select = no. of qubits must be even")
    N = jet_etaphipt.shape[0]  # Number of jets

    # Step 1: Create mask limit to determine how many particles can be selected from each subjet
    if selection=='random':
        mask_limit = (n_qubits//2)*np.ones([N,1])#np.where(num_PFCands_subleading_jet > n_qubits // 2, n_qubits // 2, n_qubits - num_PFCands_subleading_jet)  # (N,)
        pf_mask = evt_subjet_idx < mask_limit  # (N, 100)
        
    # Step 3: Ensure each jet has exactly n_qubits particles
        selected_particles = []
        print("It is now necessary to run an event loop in python")
        print("My sincere apologies")
        import time;time.sleep(3)
        for i in range(N):
            jet_particles = jet_etaphipt[i]  # (100, 3)
            mask = pf_mask[i]  # (100,)
            selected = jet_particles[mask]  # Select particles based on mask
            selected_idx=evt_subjet_idx[i][mask]
            if len(selected) < n_qubits:
                # Randomly sample additional particles to reach n_qubits
                non_padded_indices = np.where(~mask)[0]  # Indices of non-padded particles
                num_additional = n_qubits - len(selected)
                additional_indices = np.random.choice(non_padded_indices, size=num_additional, replace=False)
                additional_particles = jet_particles[additional_indices]
                selected = np.vstack((selected, additional_particles))

            selected_particles.append(selected)

        # Step 4: Convert the list to a numpy array of shape (N, n_qubits, 3)
        jet_etaphipt = np.array(selected_particles)  # (N, n_qubits, 3)
        return jet_etaphipt
    elif selection=='equal':
        mask_limit = np.where(num_PFCands_subleading_jet > n_qubits // 2, n_qubits // 2, n_qubits - num_PFCands_subleading_jet)  # (N,)
        pf_mask=evt_subjet_idx<mask_limit
        extra_mask=np.sum(pf_mask,axis=1)==n_qubits
        
        jet_etaphipt=jet_etaphipt[extra_mask] # N x n_qubits x 3 
        pf_mask=pf_mask[extra_mask]
        jet_etaphipt_selected=jet_etaphipt[pf_mask].reshape(-1,n_qubits,3) # N x 2 x n_qubits x 3
        return jet_etaphipt_selected,extra_mask    
    else:
        raise NameError("Selection must be either random or equal")
    # Step 2: Create a mask to select particles from each jet
    
    