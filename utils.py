import os
import pickle
import numpy as np

pwd= os.getcwd()





def check_dir(path):
    '''check if <path> to dir exists or not. If it doesn't, create the <dir> returns the absolute path to the created dir'''
    pwd=os.getcwd()
    try:
        os.chdir(path)
    except OSError:
        os.mkdir(path)
        os.chdir(path)
    path=os.getcwd()
    os.chdir(pwd)
    return path
    
def print_events(events,name=None):
    '''Function for printing nested dictionary with at most 3 levels, with final value being a numpy.ndarry, prints the shape of the array'''
    if name: print (name)
    for channel in events:
        if type(events[channel]) == np.ndarray or type(events[channel]) == list:
            if type(events[channel]) == np.ndarray or channel=='EventAttribute': print ("    Final State:", channel,np.array(events[channel]).shape)
            else: 
                try: print ("    Final State:", channel,[item.shape for item in events[channel]])
                except AttributeError:
                    print ("    Final State:", channel,len(events[channel]))
            continue
        print ("Channel: ",channel)
        if type(events[channel]) != dict: continue
        for topology in events[channel]:
            if type(events[channel][topology])!= dict: continue
            if type(events[channel][topology])==np.ndarray  or type(events[channel]) == list:
                print ("    Final State: ",topology, np.array(events[channel][topology]).shape)
                continue
            print ("Topology: ",topology)
            for final_state in events[channel][topology]:
                print ("    Final State: ",final_state," Shape: ",events[channel][topology][final_state].shape)
    return





def Unpickle(filename,path=None,load_path=".",verbose=True,keys=None,extension='.pickle'):
    '''load <python_object> from <filename> at location <load_path>'''
    if '.' not in filename: filename=filename+extension
    if path is not None: load_path=path
    pwd=os.getcwd()
    if load_path != ".": os.chdir(load_path)
    if filename[-4:]==".npy":
        ret=np.load(filename,allow_pickle=True)
        if verbose: print (filename," loaded from ",os.getcwd())
        os.chdir(pwd)
        return ret
    try:
        with open(filename,"rb") as File:
            return_object=pickle.load(File)
    except Exception as e:
        print (e," checking if folder with ",filename.split(".")[0]," exists..")
        try: os.chdir(filename.split(".")[0])   
        except Exception as e: 
            os.chdir(pwd)
            raise e     
        print ("exists! loading...")
        return_object=folder_load(keys=keys)
    if verbose: print (filename," loaded from ",os.getcwd())
    os.chdir(pwd)
    return return_object
def Pickle(python_object,filename,path=None,save_path=".",verbose=True,overwrite=True,append=False,extension='.pickle'):
    '''save <python_object> to <filename> at location <save_path>'''
    if '.' not in filename: filename=filename+extension
    if path is not None: save_path=path
    pwd=os.getcwd()
    if save_path != "." :
        os.chdir(save_path)
    if not overwrite:
        if filename in os.listdir("."): 
            raise IOError("File already exists!")
    if append: 
        assert type(python_object)==dict
        prev=Unpickle(filename)
        print_events(prev,name="old")
        python_object=merge_flat_dict(prev,python_object)
        print_events(python_object,name="appended")
    if type(python_object)==np.ndarray:
        np.save(filename,python_object)
        suffix=".npy"
    else:
        try:
            File=open(filename,"wb")
            pickle.dump(python_object,File)
        except OverflowError as e:
            File.close()
            os.system("rm "+filename)
            os.chdir(pwd)
            print (e,"trying to save as numpy arrays in folder...")
            folder_save(python_object,filename.split(".")[0],save_path)
            return
        suffix=""
    if verbose: print (filename+suffix, " saved at ", os.getcwd())
    os.chdir(pwd)
    return
def folder_save(events,folder_name,save_path,append=False):
    pwd=os.getcwd()
    os.chdir(save_path) 
    try: os.mkdir(folder_name)
    except FileExistsError as e: 
        print (e,"Overwriting...")
    finally:os.chdir(folder_name)                      
    for item in events: 
        if append:
            print ("appending...") 
            events[item]=np.concatenate((np.load(item+".npy",allow_pickle=True),events[item]),axis=0)
        if type(events[item]) ==list:
            print("list type found as val, creating directory...")
            os.mkdir(item)
            os.chdir(item)
            for i,array in enumerate(events[item]):
                np.save(item+str(i),array,allow_pickle=True)
                print (array.shape,"saved at ",os.getcwd())
            os.chdir("..")
        else: 
            np.save(item,events[item],allow_pickle=True)
            print (item+".npy saved at ",os.getcwd(), "shape = ",events[item].shape)
    os.chdir(pwd)
    return

def folder_load(keys=None,length=None):
    events=dict()
    pwd=os.getcwd()
    for filename in os.listdir("."):
        if os.path.isdir(filename):
            os.chdir(filename)
            events[filename]=[np.load(array_files,allow_pickle=True) for array_files in os.listdir(".")]
            os.chdir("..")  
            continue          
        if keys is not None:
            if filename[:-4] not in keys: continue
        try:
            events[filename[:-4]]=np.load(filename,allow_pickle=True)[:length]
        except IOError as e:
            os.chdir(pwd)
            raise e
        else:
            print (filename[:-4]," loaded to python dictionary...")
    return events

    
    
    
    



