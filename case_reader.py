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
class CASEDelphesJetDataset(IterableDataset):
    """
    Iterable dataset class to load jet data from .h5 files.

    Args:
        filelist (List[str]): List of file paths to the .h5 files.
        batch_size (int): Number of samples in each batch.
        max_samples (int): Maximum number of samples to load.
        data_key (str): Key to access the jet data inside the .h5 files.
        feature_key (str): Key to access event-level features.
        input_shape (tuple[int]): Input shape specifying number of particles and features.
        epsilon (float): Small constant to avoid division by zero.
        train (bool): Whether to yield only training data, or labels as well.
        yield_energies (bool): If True, yield jet energies as well.

    Yields:
        np.ndarray: Batch of data samples.
    """
    def __init__(self, filelist:List[str]=None, batch_size:int=32, max_samples:int=5e4, data_key='jetConstituentsList',\
                 feature_key='eventFeatures',input_shape:tuple[int]=(10, 3),epsilon:float=1.0e-4,train:bool=True,\
                    normalize_pt:bool=False,logger=None):
        super().__init__()
        self.filelist = sorted(filelist)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pt_index=ut.getIndex('particle','pt')
        self.eta_index=ut.getIndex('particle','eta')
        self.phi_index=ut.getIndex('particle','phi')
        self.j1pt_index=ut.getIndex('event','j1Pt')
        self.j2pt_index=ut.getIndex('event','j2Pt') # Bug fix
        self.mjj_index=ut.getIndex('event','mJJ')
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
        Loads and preprocesses a single .h5 file.

        Args:
            file_path (str): Path to the .h5 file.
            inference (bool): Whether the data is being loaded for inference.

        Returns:
            np.ndarray: Stacked data from both jets.
            np.ndarray: Stacked truth labels.
        """
        self.max_samples=2*self.max_samples # to account for 2 jets per event
        with h5py.File(file_path, 'r') as file:
            # Read data of shape (N,2,100,3) where N is the number of events, 2 is the number of jets, 100 is the number of particles and 3 is the (eta,phi,pt) of each particle
            mjj=np.array(file[self.feature_key][:,self.mjj_index])
            j1pt=np.array(file[self.feature_key][:,self.j1pt_index])
            j2pt=np.array(file[self.feature_key][:,self.j2pt_index])

            
            if self.logger is not None:
                self.logger.info(f"Reading hardest {self.n_qubits} PFCands per jet")
            else:
                print(f"Reading hardest {self.n_qubits} PFCands per jet")
            jet_etaphipt = np.array(file[self.data_key][()]) # because n_qubits is the number of particles
            sorted_indices = np.argsort(-jet_etaphipt[...,self.pt_index], axis=2)
            jet_etaphipt = np.take_along_axis(jet_etaphipt, sorted_indices[...,None], axis=2)
            jet_etaphipt = jet_etaphipt[:,:,:self.n_qubits,:]
            #jet1_energy=np.array(file[self.feature_key][:,self.j1E_index])
            #jet2_energy=np.array(file[self.feature_key][:,self.j2E_index])
            
            try:
                truth_label = np.array(file['truth_label'][()])
            except:
                if 'qcd'.casefold() in file_path.casefold(): # case-insensitive search
                    truth_label = np.zeros(jet_etaphipt.shape[0])
                else:
                    truth_label = np.ones(jet_etaphipt.shape[0])
        if self.normalize_pt:
            print("Normalizing PFCand pT by jet pT")
            jet_etaphipt[:,0,:,self.pt_index]=jet_etaphipt[:,0,:,self.pt_index]/j1pt[:,np.newaxis]
            jet_etaphipt[:,1,:,self.pt_index]=jet_etaphipt[:,1,:,self.pt_index]/j2pt[:,np.newaxis]
        else:
            jet_etaphipt[:,0,:,self.pt_index]=self.fixed_rescale(jet_etaphipt[:,0,:,self.pt_index], epsilon=self.epsilon,type='pt')
            jet_etaphipt[:,1,:,self.pt_index]=self.fixed_rescale(jet_etaphipt[:,1,:,self.pt_index], epsilon=self.epsilon,type='pt')
        
        jet_etaphipt[:,0,:,self.eta_index]=self.fixed_rescale(jet_etaphipt[:,0,:,self.eta_index], epsilon=self.epsilon,type='eta')
        jet_etaphipt[:,0,:,self.phi_index]=self.fixed_rescale(jet_etaphipt[:,0,:,self.phi_index], epsilon=self.epsilon,type='phi')
        jet_etaphipt[:,1,:,self.eta_index]=self.fixed_rescale(jet_etaphipt[:,1,:,self.eta_index], epsilon=self.epsilon,type='eta')
        jet_etaphipt[:,1,:,self.phi_index]=self.fixed_rescale(jet_etaphipt[:,1,:,self.phi_index], epsilon=self.epsilon,type='phi')
            
        # If you only wish to test, then no need to batch, just return the entire array
        if inference:
            return jet_etaphipt,np.stack([mjj,j1pt,j2pt],axis=-1),truth_label
        
        stacked_data = np.reshape(jet_etaphipt,newshape=[-1, jet_etaphipt.shape[2], jet_etaphipt.shape[3]])
        stacked_labels = np.concatenate([truth_label, truth_label], axis=0)
        
        print("sample max pt: ",np.max(stacked_data[:,:,self.pt_index]))
        print("sample min pt: ",np.min(stacked_data[:,:,self.pt_index]))
        print("sample max eta: ",np.max(stacked_data[:,:,self.eta_index]))
        print("sample min eta: ",np.min(stacked_data[:,:,self.eta_index]))
        print("sample max phi: ",np.max(stacked_data[:,:,self.phi_index]))
        print("sample min phi: ",np.min(stacked_data[:,:,self.phi_index]))
        return stacked_data, stacked_labels#,stacked_energies
    
    def load_for_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
    Loads data from the specified .h5 files for inference.

    This method processes the jet and event data from the files, ensuring that 
    only a maximum number of samples specified by `max_samples` are loaded. 
    It returns separate arrays for the jet features of the two jets in each event,
    the dijet invariant mass (mjj), and the corresponding truth labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - jet_array[:,0,:,:] (np.ndarray): Jet features for the first jet in each event, with shape (N, num_particles, 3).
            - jet_array[:,1,:,:] (np.ndarray): Jet features for the second jet in each event, with shape (N, num_particles, 3).
            - jetFeatures_array (np.ndarray): (mjj,j1pt,j2pt) values for each event, with shape (N,3).
            - truth_labels (np.ndarray): Ground truth labels for each event, with shape (N,).

    Notes:
        - This method halves the number of samples from `max_samples` to avoid double counting, because we wish to return 2 jets per event
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
        return jet_array[:,0,:,:],jet_array[:,1,:,:],jetFeatures_array,truth_labels

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

def OneP1QDataLoader(input_shape:tuple[int]=(100, 3),train:bool=True,dataset='delphes',**kwargs) -> DataLoader:
    '''
    Wrapper function to create a DataLoader for the CASEDelphesJetDataset.
    Args:
        filelist (List[str]): List of file paths to the .h5 files containing the data.
        batch_size (int): Number of samples in each batch.
        shuffle_buffer_size (int): Number of samples to shuffle in the buffer.
        input_shape (tuple[int]): Shape of the input data, with the first element being the no. of particles per jet to read and the 2nd typically being (eta,phi,pt).
    Returns:
        DataLoader: Torch DataLoader object that yields batches of data.
    '''
    print(f"Will read only {input_shape[0]} particles per jet")
    
    
    if dataset.casefold()=='delphes':
        dset = CASEDelphesJetDataset(input_shape=input_shape,train=train,feature_key='eventFeatures',**kwargs)
    elif dataset.casefold()=='jetclass':
        dset = CASEJetClassDataset(input_shape=input_shape,train=train,feature_key='jetFeatures',**kwargs)
    else:
        raise NameError("Dataset type must be either delphes or jetclass")
    return DataLoader(dset, batch_size=None)  # None for batch_size since batching is managed by the dataset



class CASEJetClassDataset(CASEDelphesJetDataset):
    def __init__(self,feature_key='jetFeatures',selection='equal',**kwargs):
        super().__init__(feature_key=feature_key,**kwargs)
        self.jpt_index=ut.getIndex('jet','jet_pt')
        self.j_msd_index=ut.getIndex('jet','jet_sdmass')
        self.selection=selection
    def load_and_preprocess_file(self, file_path:str,inference:bool=False):
        """
        Loads and preprocesses a single .h5 file.

        Args:
            file_path (str): Path to the .h5 file.
            inference (bool): Whether the data is being loaded for inference.

        Returns:
            np.ndarray: Stacked data from a jet.
            np.ndarray: Stacked truth labels.
        """
        with h5py.File(file_path, 'r') as file:
            # Read data of shape (N,2,100,3) where N is the number of events, 2 is the number of jets, 100 is the number of particles and 3 is the (eta,phi,pt) of each particle
            msd=np.array(file[self.feature_key][:,self.j_msd_index])
            jet_pt=np.array(file[self.feature_key][:,self.jpt_index])
            jet_features=np.array(file[self.feature_key])
            
            if self.logger is not None:
                self.logger.info(f"Reading hardest {self.n_qubits} PFCands per jet")
            else:
                print(f"Reading hardest {self.n_qubits} PFCands per jet")
            jet_etaphipt = np.array(file[self.data_key][()]) # because n_qubits is the number of particles
            sorted_indices = np.argsort(-jet_etaphipt[...,self.pt_index], axis=-1)
            jet_etaphipt = np.take_along_axis(jet_etaphipt, sorted_indices[...,None], axis=1)
            jet_etaphipt = jet_etaphipt[:,:self.n_qubits,:]
                
            try:
                truth_label = np.array(file['truth_labels'][()])
            except:
                if 'zjetsto'.casefold() in file_path.casefold(): # case-insensitive search
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
            
        # If you only wish to test, then no need to batch, just return the entire array
        
        print("sample max pt: ",np.max(jet_etaphipt[:,:,self.pt_index]))
        print("sample min pt: ",np.min(jet_etaphipt[:,:,self.pt_index]))
        print("sample max eta: ",np.max(jet_etaphipt[:,:,self.eta_index]))
        print("sample min eta: ",np.min(jet_etaphipt[:,:,self.eta_index]))
        print("sample max phi: ",np.max(jet_etaphipt[:,:,self.phi_index]))
        print("sample min phi: ",np.min(jet_etaphipt[:,:,self.phi_index]))
        if inference:
            return jet_etaphipt,jet_features,truth_label
        return jet_etaphipt, truth_label#,stacked_energies
    
    def load_for_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
    Loads data from the specified .h5 files for inference.

    This method processes the jet and event data from the files, ensuring that 
    only a maximum number of samples specified by `max_samples` are loaded. 
    It returns separate arrays for the jet features of the two jets in each event,
    the dijet invariant mass (mjj), and the corresponding truth labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - jet_array (np.ndarray): Jet PFCand features, with shape (N, num_particles, 3).
            - jetFeatures_array (np.ndarray): (mjj,j1pt,j2pt) values for each event, with shape (N,3).
            - truth_labels (np.ndarray): Ground truth labels for each event, with shape (N,).
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
    
    