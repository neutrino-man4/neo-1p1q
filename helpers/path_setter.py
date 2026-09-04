import os

def getIndex(which:str='particle',feat:str=None)->int:
    if which=='particle':
        nameArray=particleFeatureNames
    elif which=='event':
        nameArray=eventFeatureNames
    else:
        print("arg which must be either event or particle. Returning -1 as index")
        return -1
    try:
        idx=nameArray.index(feat)
    except ValueError as v:
        print(f"item {feat} not found in list {nameArray}")
        idx=-1
    return idx

class PathSetter:
    def __init__(self,data_path:str=None):
        self.data_path=data_path
    def get_data_path(self,key:str=None)->str:
        if key is None:
            raise KeyError("Where is the damn key?")
        if key not in path_dict.keys():
            print("You got the wrong key")
            raise KeyError(f"Key {key} not found in {path_dict.keys()}")
            
        return os.path.join(self.data_path,path_dict[key])


path_dict:dict[str:str]={'QCD_train':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/',
        'QCD_test':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/test/',
        'QCD_SR':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/',
        'QCD_lib':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/library/',
        'QCD_flat_mjj':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/flat_train/',
        'QCD_flat_pt':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/flat_train_j1pt/',
        'QCD_SR_train_flat':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/flat_train/',
        'QCD_SR_train':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/train/',
        'QCD_SR_test':'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/test/',
        'grav_1p5_narrow':'grav_1p5_na/',
        'grav_2p5_narrow':'grav_2p5_na/',
        'grav_3p5_narrow':'grav_3p5_na/',
        'grav_4p5_narrow':'grav_4p5_na/',
        'grav_3p5_broad':'grav_3p5_br/',
        'grav_2p5_broad':'grav_2p5_br/',
        'grav_1p5_broad':'grav_1p5_br/',
        'AtoHZ_1p5':'AtoHZ_1p5/',
        'AtoHZ_2p5':'AtoHZ_2p5/',
        'AtoHZ_3p5':'AtoHZ_3p5/',
        'AtoHZ_4p5':'AtoHZ_4p5/',
        'ZJetsToNuNu_flat':'flat_train/ZJetsToNuNu/',
        'ZJetsToNuNu_train':'train/ZJetsToNuNu/',
        'ZJetsToNuNu_test':'test/ZJetsToNuNu/',
        'ZJetsToNuNu_val':'val/ZJetsToNuNu/',
        'HToCC': 'train/HToCC/',
        'HToBB': 'train/HToBB/',
        'WToQQ': 'train/WToQQ/',
        'TTBarLep': 'train/TTBarLep/',
        'TTBar': 'train/TTBar_/',
        'HToGG': 'train/HToGG/',
        'ZToQQ': 'train/ZToQQ/',
        'TTBar_flat_train':'flat_train/TTBar_/',
        'ZJetsToNuNu_flat_test':'flat_test/ZJetsToNuNu/',
        'HToCC_flat_test':'flat_test/HToCC/',
        'HToBB_flat_test':'flat_test/HToBB/',
        'WToQQ_flat_test':'flat_test/WToQQ/',
        'TTBarLep_flat_test':'flat_test/TTBarLep/',
        'TTBar_flat_test':'flat_test/TTBar_/',
        'HToGG_flat_test':'flat_test/HToGG/',
        'ZToQQ_flat_test':'flat_test/ZToQQ/',
        'VQC_train':'VQC/merged/TTBar_QCD/flat_train',
        'VQC_test':'VQC/merged/TTBar_QCD/flat_test/',
        'VQC_val':'VQC/merged/TTBar_QCD/flat_val/',
           }

eventFeatureNames:list[str]=['mJJ', 'j1Pt', 'j1Eta', 'j1Phi', 'j1M', 'j1E', 'j2Pt',
       'j2M', 'j2E', 'DeltaEtaJJ', 'DeltaPhiJJ']

particleFeatureNames:list[str]=['eta', 'phi', 'pt']

labels={'grav_1p5_narrow':'$M_{grav}=1.5$ TeV','grav_2p5_narrow':'$M_{grav}=2.5$ TeV','grav_3p5_narrow':'$M_{grav}=3.5$ TeV'\
        ,'grav_4p5_narrow':'$M_{grav}=4.5$ TeV','AtoHZ_1p5':'$M_{A}=1.5$ TeV','AtoHZ_2p5':'$M_{A}=2.5$ TeV',\
            'AtoHZ_3p5':'$M_{A}=3.5$ TeV','AtoHZ_4p5':'$M_{A}=4.5$ TeV','ZJetsToNuNu':'q/g jets','HToCC':r'$H \rightarrow c \overline{c}$',\
                'HToBB':r'$H \rightarrow b \overline{b}$',\
                    'WToQQ':r'$W \rightarrow q \overline{q}$','TTBarLep':r'$t \rightarrow b l \nu$',\
                        'HToGG':r'$H \rightarrow gg$','TTBar':r'$t \rightarrow bq \overline{q}$',\
                            'ZToQQ':r'$Z \rightarrow q \overline{q}$',\
                                'TTBarLep':r'$t \rightarrow bl\nu$',\
                                    'TTBar':r'$t \rightarrow bq\overline{q}$',
                                    'VQC_test':r'q/g jets vs $t \rightarrow bq\overline{q}'}