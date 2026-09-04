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
base_path='/ceph/abal/QML/delphes/substructure/debug/'
#base_path='/ceph/abal/QML/CMS_MC/substructure/debug/'

signals=['AtoHZ_1p5','AtoHZ_2p5','AtoHZ_3p5','AtoHZ_4p5']
#signals=['grav_1p5_na','grav_2p5_na','grav_3p5_na','grav_4p5_na']#rrow','grav_1p5_broad','grav_2p5_broad','grav_3p5_broad']

files={}
files['qcd_unflattened_substructure']=sorted(glob.glob('/ceph/abal/QML/delphes/substructure/debug/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/*.h5'))[0]
#files['qcd_flattened_substructure']=sorted(glob.glob('/ceph/abal/QML/delphes/substructure/debug/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/flat_train/*.h5'))[0]
#files['qcd_flattened']=sorted(glob.glob(base_path+'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/flat_train/*.h5'))[0]
#files['qcd_unflattened']=sorted(glob.glob(base_path+'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/*.h5'))[0]


### FOR CMS MC ####
# signals=['YtoHH_Htott_Y2000_H400','YtoHH_Htott_Y3000_H400','YtoHH_Htott_Y5000_H400']
# #signals=[os.path.split(k)[-1] for k in glob.glob('/ceph/abal/QML/CMS_MC/substructure/debug/XToYY*400*')]
# files['qcd_unflattened_substructure']=sorted(glob.glob('/ceph/abal/QML/CMS_MC/substructure/debug/qcd_sideband/*.h5'))[0]

for signal in signals:
    files[signal]=sorted(glob.glob(base_path+f'{signal}/*.h5'))[0]

ipath='plots/sideband/exclusive_CA/delphes/'
pathlib.Path(ipath+f'PFCand_wise/').mkdir(parents=True,exist_ok=True)
        
#files['qcd_unflattened']=sorted(glob.glob(base_path+'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/*.h5'))[0]

if 'AtoHZ' in signals[0]: sig_id='AtoHZ'
elif 'XToYY' in signals[0]: sig_id='XToYY'
elif 'grav' in signals[0]: sig_id='grav'
else: sig_id='YtoHH'

bin_nums={'pt':60,'eta':62,'phi':62}
range_limits={'pt':(0,0.3),'eta':(-np.pi,np.pi),'phi':(-np.pi,np.pi)}
labels={'pt':'$p_T$','eta':'$\eta$','phi':'$\phi$'}
def get_data(filepath,Q=7):
    with h5py.File(filepath, 'r') as file:
        print(filepath)
        try:
            pf_pt=file['jetConstituentsList'][...,2]
            pf_eta=file['jetConstituentsList'][...,0]
            pf_phi=file['jetConstituentsList'][...,1]
        except:
            pf_pt=file['particleFeatures'][...,2]
            pf_eta=file['particleFeatures'][...,0]
            pf_phi=file['particleFeatures'][...,1]
        num_PFCands_subleading_jet=file['num_PFCands_subleading_jet'][()]
        pf_pt_scaled=(pf_pt-0.0)/(3000.0-0.0) 
        pf_eta_scaled=(pf_eta+0.8)/(1.6)*(2*np.pi)-np.pi
        pf_phi_scaled=(pf_phi+0.8)/(1.6)*(2*np.pi)-np.pi
        pf_feature_scaled=np.stack([pf_pt_scaled,pf_eta_scaled,pf_phi_scaled],axis=-1)
        j1pt=file['eventFeatures'][:,1] # For CMS MC, make this index equal to 2
        j2pt=file['eventFeatures'][:,6]
        mjj=file['eventFeatures'][:,0]

        
        
        NUM_SELECTED_PFCANDS=qubits
        
        
        
        evt_subjet_idx=file['PFCand_subjet_idx'][()]
    mask_limit=np.where(num_PFCands_subleading_jet>NUM_SELECTED_PFCANDS//2,NUM_SELECTED_PFCANDS//2,NUM_SELECTED_PFCANDS-num_PFCands_subleading_jet)
    pf_mask=evt_subjet_idx<mask_limit[...,None]
    
    hmm_mask=np.all(np.sum(pf_mask,axis=2)==NUM_SELECTED_PFCANDS,axis=1)
    pf_mask=pf_mask[hmm_mask]
    
    j1pt=j1pt[hmm_mask]
    j2pt=j2pt[hmm_mask]
    pf_pt=pf_pt[hmm_mask]
    num_PFCands_subleading_jet=num_PFCands_subleading_jet[hmm_mask]
    
    pf_pt=pf_pt[pf_mask].reshape(-1,2,NUM_SELECTED_PFCANDS)
    pf_pt_norm=pf_pt

    pf_pt_norm[:,0,:]=pf_pt[:,0,:]/j1pt[:,np.newaxis]
    pf_pt_norm[:,1,:]=pf_pt[:,1,:]/j2pt[:,np.newaxis]
    #import pdb;pdb.set_trace()
    pf_feature_scaled=pf_feature_scaled[hmm_mask]
    return pf_feature_scaled[:,:,:Q,:],pf_pt_norm,num_PFCands_subleading_jet
    
        

bins={}
edges={}
nbins={}
nedges={}
pf_feature_scaled={}
pf_eta_scaled={}
pf_phi_scaled={}

pf_pt_norm={}
num_PFCands_subleading={}
for file,filepath in files.items():
    pf_feature_scaled[file],pf_pt_norm[file],num_PFCands_subleading[file]=get_data(filepath,Q=qubits)
    bins,edges=np.histogram(num_PFCands_subleading[file],bins=50,range=(0,50),density=True)
    plt.stairs(bins,edges,label=file,fill=False)
plt.legend(loc='upper right')
plt.title('Number of subleading PFCands')
plt.xlabel('Number of subleading PFCands')
plt.ylabel('Density')
plt.savefig(ipath+f'PFCand_wise/num_PFCands_subleading_{sig_id}.png')

for f,feature in enumerate(['pt','eta','phi']):
    plt.clf()    
    for Q in range(qubits):
        maxi=2
        if Q>3:maxi=0.5
        if Q>0:plt.clf()
        fill=False
        alpha=0.5
        
        for file in pf_feature_scaled.keys():
            bins,edges=np.histogram(pf_feature_scaled[file][:,:,Q,f].flatten(),bins=bin_nums[feature],range=range_limits[feature],density=True)
            i=0
            if 'qcd' in file:
                fill=True
                i+=0.2
            else:
                alpha=1.
                i=0.8
                fill=False
                
            #import pdb;pdb.set_trace()
            plt.stairs(bins,edges,label=file,fill=fill,alpha=0.2+i)

        plt.title('Scaled assuming maximum of 3000 GeV')
        plt.xlabel(f'{labels[feature]} (hardest PFCand ID: {Q} [GeV])')
        #plt.xlabel(f'$p_T$ ({Q} hardest PFCands [GeV])')
        plt.legend(loc='upper right')
        plt.savefig(ipath+f'PFCand_wise/{feature}_scaled_{sig_id}_PFCand{Q}.png')
        #plt.savefig(ipath+f'./feature_scaled_{sig_id}.png')

        fill=False

        plt.clf()
        if feature=='pt':
            for file in pf_pt_norm.keys():
                
                nbins,nedges=np.histogram(pf_pt_norm[file][...,Q].flatten(),bins=199,range=(0,maxi),density=True)
                i=0
                if 'qcd' in file:
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
            plt.savefig(ipath+f'PFCand_wise/pt_norm_{sig_id}_PFCand{Q}.png')