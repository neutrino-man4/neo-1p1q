import os

def getIndex(which:str='particle',feat:str=None)->int:
    if which=='particle':
        nameArray=particleFeatureNames
    else:
        print("arg which must be 'particle'. Returning -1 as index")
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


# JetClass layout: <data_dir>/<split>/<sample>/<sample>_NNN.h5.
# The VQC_* keys point at a whole split; callers glob it recursively for *.h5.
path_dict:dict[str:str]={
        'VQC_train':'train/',
        'VQC_val':'val/',
        'VQC_test':'test/',
        'VQC_train_flat':'flat_train/',
        'VQC_val_flat':'flat_val/',
        'VQC_test_flat':'flat_test/',
        'ZJetsToNuNu_train':'train/ZJetsToNuNu/',
        'ZJetsToNuNu_val':'val/ZJetsToNuNu/',
        'ZJetsToNuNu_test':'test/ZJetsToNuNu/',
        'HToCC': 'train/HToCC/',
        'HToBB': 'train/HToBB/',
        'WToQQ': 'train/WToQQ/',
        'TTBarLep': 'train/TTBarLep/',
        'TTBar': 'train/TTBar_/',
        'HToGG': 'train/HToGG/',
        'ZToQQ': 'train/ZToQQ/',
           }

particleFeatureNames:list[str]=['eta', 'phi', 'pt']

labels={'ZJetsToNuNu':'q/g jets','HToCC':r'$H \rightarrow c \overline{c}$',\
                'HToBB':r'$H \rightarrow b \overline{b}$',\
                    'WToQQ':r'$W \rightarrow q \overline{q}$','TTBarLep':r'$t \rightarrow bl\nu$',\
                        'HToGG':r'$H \rightarrow gg$',\
                            'ZToQQ':r'$Z \rightarrow q \overline{q}$',\
                                'TTBar':r'$t \rightarrow bq\overline{q}$',
                                    'VQC_test':r'q/g jets vs $t \rightarrow bq\overline{q}$'}