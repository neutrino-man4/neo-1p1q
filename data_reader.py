import os,sys
from utils import Unpickle


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import print_events
import __main__





def scale_feature(X,epsilon=0.,max_array=None,min_array=None):
    scaler =  MinMaxScaler(feature_range = (0 , np.pi-epsilon))
    if max_array is None:
        X = scaler.fit_transform(X)
        passed=np.arange(len(X))
    else: 
        assert min_array is not None
        passed_inds=[]
        for array,M in zip(np.swapaxes(X,0,1),max_array):
            passed_inds.append(array<M)
        passed_inds=np.array(passed_inds)
        passed_inds=np.sum(passed_inds,axis=0)
        rejected=np.where(passed_inds != X.shape[-1])[0]
        passed_inds=passed_inds==X.shape[-1]
        X=X[passed_inds]
        X=np.concatenate(([min_array],X,[max_array]),axis=0)
        X=scaler.fit_transform(X)[1:-1]

    return X,passed_inds     
def get_data(**kwargs):
    unscaled=kwargs.get("unscaled",False)
    main_filename=__main__.__file__.split('/')[-1]
    if unscaled: assert main_filename!='auto_qml.py'

    max_values={'lep1pt':1000, 
                'lep2pt':900, 
                'theta_ll':np.pi,
                'b1pt':1000, 
                'b2pt':900,
                'theta_bb':np.pi, 
                'MET':1000
                }
    min_values={'lep1pt':0., 
                'lep2pt':0., 
                'theta_ll':0.,
                'b1pt':0., 
                'b2pt':0.,
                'theta_bb':0., 
                'MET':0.,
                } # MINIMUM OF NON PERIODIC VARIABLES SET TO 0 FOR PRESERVING TOPOLOGY
    combined_signal=kwargs.get('combined_signal',False)
    all_files=['sample_bg_morestat.csv','sample_mH2500_morestat.csv']
    #'sample_bg.csv','sample_mH500.csv','sample_mH1000.csv','sample_mH1500.csv','sample_mH2000.csv','sample_mH2500.csv'
    csv_dir = './data/'
    keys=kwargs.get('keys')
    max_array=[]
    min_array=[]
    for item in keys:
        min_array.append(min_values[item])
        max_array.append(max_values[item])
    min_array,max_aray=np.array(min_array),np.array(max_array)
    assert keys is not None,'Provide compulsory kwarg keys=<list of df keys>'
    train=kwargs.get('train',True)
    signal_files=[]
    bg_filename=kwargs.get('train_file')#'sample_bg.csv'
    signal_files=all_files+[]
    assert bg_filename in all_files,f"Provide compulsory kwarg: train_file in\n {all_files}"
    signal_files.remove(bg_filename)
    bkg = pd.read_csv(csv_dir+bg_filename,delimiter=',')
    unscaled_bg_array=bkg[keys].values
    #print ('Max before scaling:',np.max(unscaled_bg_array,axis=0),np.min(unscaled_bg_array,axis=0))
    bg_array,passed=scale_feature(unscaled_bg_array,max_array=max_array,min_array=min_array)
    unscaled_bg_array=unscaled_bg_array[passed]
    if unscaled:
        bg_array=unscaled_bg_array
    #print ('Max after scaling:',np.max(unscaled_bg_array,axis=0),np.min(unscaled_bg_array,axis=0))
    if 'bg' in bg_filename:
        train_X=bg_array[:30000]
        val_X=bg_array[30000:35000]
        test_X=bg_array[35000:50000]
    else:
        length=len(bg_array)
        train_X=bg_array[:int(0.5*length)]
        val_X=bg_array[int(0.5*length):int(0.75*length)]
        test_X=bg_array[int(0.75*length):]
    if train:
        return {
                'train':{'X':train_X,'y':np.zeros(len(train_X))},
                'val':{'X':val_X,'y':np.zeros(len(val_X))},
               }
    return_dict={bg_filename:test_X}          
    for item in signal_files:
        print ('\nLoading :',item)
        events=pd.read_csv(csv_dir+item,delimiter=',')
        unscaled_values=events[keys].values   
        
        values,passed=scale_feature(unscaled_values,max_array=max_array,min_array=min_array)
        unscaled_values=unscaled_values[passed]
       
        if unscaled: return_dict[item]=unscaled_values
        else: return_dict[item]=values
    return_dict['train_filename']=bg_filename
    
    if not combined_signal:
        print_events(return_dict,name='test dictionary') 
        return return_dict
    X,y,y_map=[],[],{}
    i=0
    for key,val in return_dict.items():
        if key=='train_filename': continue
        X.append(val)
        y_map[i]=key
        y.append(np.full(len(val),i))
    test_dict={'X':np.concatenate(X,axis=0),'y':np.concatenate(y,axis=0),'y_map':y_map,'train_filename':bg_filename}
    print_events(test_dict,name='test_dict')
    return test_dict    
 
