import os;os.environ['PYTHONPATH']=os.environ['QML']
import matplotlib.pyplot as plt
import numpy as np
import matplotlib;matplotlib.use('Agg')
import h5py,glob
import helpers.path_setter as ps
from sklearn.metrics import roc_curve, auc
import pathlib
import mplhep as hep
hep.style.use("CMS")

plt.rcParams["figure.figsize"] = (8,8)
qubits=8
# base_path='/storage/9/abal/CASE/delphes/'
# base_path='/ceph/abal/QML/delphes/substructure/debug/'
base_path='/ceph/abal/QML/JetClass/train/ZJetsToNuNu/'


#signals=['grav_1p5_na','grav_2p5_na','grav_3p5_na','grav_4p5_na']#rrow','grav_1p5_broad','grav_2p5_broad','grav_3p5_broad']

files={}
#files['qcd_unflattened_substructure']=sorted(glob.glob('/ceph/abal/QML/delphes/substructure/debug/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/*.h5'))[0]
#files['qcd_flattened_substructure']=sorted(glob.glob('/ceph/abal/QML/delphes/substructure/CA_decluster/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/flat_train/*.h5'))[0]
#files['qcd_flattened']=sorted(glob.glob(base_path+'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/flat_train/*.h5'))[0]
#files['qcd_unflattened']=sorted(glob.glob(base_path+'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/*.h5'))[0]


### FOR CMS MC ####
#signals=['YtoHH_Htott_Y2000_H400','YtoHH_Htott_Y3000_H400','YtoHH_Htott_Y5000_H400']
signals=[os.path.split(k)[-1].replace('_000.h5','') for k in glob.glob(base_path+'*.h5')]

for signal in signals:
    files[signal]=sorted(glob.glob(base_path+f'{signal}*.h5'))[0]
ipath='plots/JetClass/'
pathlib.Path(ipath+f'PFCand_wise/').mkdir(parents=True,exist_ok=True)
        
#files['qcd_unflattened']=sorted(glob.glob(base_path+'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/*.h5'))[0]


bin_nums={'pt':60,'eta':62,'phi':62}
range_limits={'pt':(0,0.3),'eta':(-np.pi,np.pi),'phi':(-np.pi,np.pi)}
labels={'pt':'$p_T$','eta':'$\eta$','phi':'$\phi$'}
def get_data(filepath,Q=8):
    with h5py.File(filepath, 'r') as file:
        print(filepath)
        pf_pt=file['jetConstituentsList'][...,2]
        pf_eta=file['jetConstituentsList'][...,0]
        pf_phi=file['jetConstituentsList'][...,1]
        pfc=file['jetConstituentsList'][()]
        pf_pt_scaled=(pf_pt-0.0)/(1500.0-0.0) 
        pf_eta_scaled=(pf_eta+0.8)/(1.6)*(2*np.pi)-np.pi
        pf_phi_scaled=(pf_phi+0.8)/(1.6)*(2*np.pi)-np.pi
        pf_feature_scaled=np.stack([pf_pt_scaled,pf_eta_scaled,pf_phi_scaled],axis=-1)
        jpt=file['jetFeatures'][:,0]
        
        NUM_SELECTED_PFCANDS=qubits
        num_PFCands_subleading_jet=file['num_PFCands_subleading_jet'][()]
    
        subjet_labels=file['subjet_labels'][()]
        evt_subjet_idx=file['PFCand_subjet_idx'][()]
    
    pfc,num_PFCands_subleading_jet=select_subjet_constituents(num_PFCands_subleading_jet, evt_subjet_idx, pfc, n_qubits=Q,selection='random')
    pf_pt=pfc[:,:,2] 
    pf_pt_norm=pf_pt/jpt[:,np.newaxis]
    
    return pf_feature_scaled[:,:Q,:],pf_pt_norm,num_PFCands_subleading_jet
        

bins={}
edges={}
nbins={}
nedges={}
pf_feature_scaled={}
pf_eta_scaled={}
pf_phi_scaled={}
num_PFCands_subleading={}
pf_pt_norm={}
mask={}
for file,filepath in files.items():
    pf_feature_scaled[file],pf_pt_norm[file],num_PFCands_subleading[file]=get_data(filepath,Q=qubits)
    bins,edges=np.histogram(num_PFCands_subleading[file],bins=30,range=(0,30),density=True)
    plt.stairs(bins,edges,label=file,fill=False)
plt.legend(loc='upper right')
plt.title('Number of subleading PFCands')
plt.xlabel('Number of subleading PFCands')
plt.ylabel('Density')
plt.savefig(ipath+f'PFCand_wise/num_PFCands_subleading.png')

for f,feature in enumerate(['pt','eta','phi']):
    plt.clf()    
    for Q in range(qubits):
        maxi=2
        if Q>3:maxi=0.5
        if Q>0:plt.clf()
        fill=False
        alpha=0.5
        
        for file in pf_feature_scaled.keys():
            bins,edges=np.histogram(pf_feature_scaled[file][:,Q,f].flatten(),bins=bin_nums[feature],range=range_limits[feature],density=True)
            i=0
            if 'qcd' in file:
                fill=True
                i+=0.2
            else:
                alpha=1.
                i=0.8
                fill=False
                
            #import pdb;pdb.set_trace()
            print('signal=',file)
            plt.stairs(bins,edges,label=file,fill=fill,alpha=0.2+i)

        plt.title('Scaled assuming maximum of 3000 GeV')
        plt.xlabel(f'{labels[feature]} (hardest PFCand ID: {Q} [GeV])')
        #plt.xlabel(f'$p_T$ ({Q} hardest PFCands [GeV])')
        plt.legend(loc='upper right')
        plt.savefig(ipath+f'PFCand_wise/{feature}_scaled_PFCand{Q}.png')
        #plt.savefig(ipath+f'./feature_scaled_{sig_id}.png')

        fill=False

        plt.clf()
        if feature=='pt':
            for file in pf_pt_norm.keys():
                
                nbins,nedges=np.histogram(pf_pt_norm[file][...,Q].flatten(),bins=199,range=(0,maxi),density=True)
                i=0
                if 'ZJets' in file:
                    fill=True
                    i+=0.2
                else:
                    alpha=1.
                    i=0.8
                    fill=False
                plt.stairs(nbins,nedges,label=file,fill=fill,alpha=0.2+i)
                #if Q>3:plt.yscale('log')
            plt.title('Normalised using $1/j_{p_T}$')
            plt.xlabel(f'$p_T$ (hardest PFCand ID: {Q} [GeV])')
            #plt.xlabel(f'$p_T$ ({qubits} hardest PFCands [GeV])')
            plt.legend(loc='upper right')
            #plt.savefig(ipath+f'./feature_norm_{sig_id}.png')
            plt.savefig(ipath+f'PFCand_wise/pt_norm_PFCand{Q}.png')



def select_subjet_constituents(num_PFCands_subleading_jet, evt_subjet_idx, jet_etaphipt, n_qubits=8,selection='equal'):
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
        jet_etaphipt_selected = np.array(selected_particles)  # (N, n_qubits, 3)

    elif selection=='equal':
        mask_limit = np.where(num_PFCands_subleading_jet > n_qubits // 2, n_qubits // 2, n_qubits - num_PFCands_subleading_jet)  # (N,)
        pf_mask=evt_subjet_idx<mask_limit
        extra_mask=np.sum(pf_mask,axis=1)==n_qubits
        import pdb;pdb.set_trace()
        jet_etaphipt=jet_etaphipt[extra_mask] # N x n_qubits x 3 
        pf_mask=pf_mask[extra_mask]
        num_PFCands_subleading_jet=num_PFCands_subleading_jet[extra_mask]
        jet_etaphipt_selected=jet_etaphipt[pf_mask].reshape(-1,n_qubits,3) # N x 2 x n_qubits x 3

    else:
        raise NameError("Selection must be either random or equal")
    # Step 2: Create a mask to select particles from each jet
    
    return jet_etaphipt_selected,num_PFCands_subleading_jet