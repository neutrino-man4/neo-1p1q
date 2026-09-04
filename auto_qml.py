



from argparse import ArgumentParser
parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--train',default=False,action='store_true',help='train network!')
parser.add_argument('--wires',default=4,type=int,help='number of wires/qubits that the circuit needs to process(AB system)')
parser.add_argument('--trash-qubits',default=1,type=int,help='number of qubits defining the B system, or the reference and trash states!')
parser.add_argument('--shots',default=5000,type=int)
parser.add_argument('-b','--batch-size',default=1,type=int)
parser.add_argument('-e','--epochs',default=20,type=int)
parser.add_argument('--backend',default='autograd')
parser.add_argument('--train-size',default=100,type=int)
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--save',default=False,action='store_true')
parser.add_argument('--save-dir',default='network_runs')
parser.add_argument('--train-file',default='sample_bg_morestat.csv')
parser.add_argument('--test',default=False,action='store_true')
parser.add_argument('--path',default='',type=str)
parser.add_argument('--keys',type=str,default=['lep1pt','lep2pt','b1pt','MET'],metavar='N',nargs='+',help='keys for training with new data')
parser.add_argument('--test-device',default='default.qubit')
args=parser.parse_args()
from utils import check_dir,print_events,Pickle,Unpickle
import os,sys

if args.save: 
    check_dir(args.save_dir)
    if args.keys[0]=='all':
        args.save_dir=check_dir(os.path.join(args.save_dir,'all_keys'))
        args.keys=['lep1pt', 'lep2pt', 'theta_ll', 'b1pt', 'b2pt','theta_bb', 'MET']
    else:
        args.keys.sort()
        args.save_dir=check_dir(os.path.join(args.save_dir,'_'.join(args.keys)))
    args.save_dir=check_dir(os.path.join(args.save_dir,'train_'+str(args.train_size)))
    num=len(os.listdir(args.save_dir))+1
    args.save_dir=os.path.join(os.getcwd(),args.save_dir,f'run_{num}')
    check_dir(args.save_dir)
    print (f'Saving run to: {args.save_dir}')

    



if len(args.keys) != args.wires:
    args.wires=len(args.keys)
    if not args.test: print (args)
    print ('Changed wires to accomodate input data!\n Ctrl+C to quit!')
    import time
    time.sleep(0.2)
args.non_trash=args.wires-args.trash_qubits
assert args.non_trash>0,'Need strictly positive dimensional compressed representation of input state!'





import pennylane.numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from itertools import combinations

import pennylane as qml
from tqdm import tqdm
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers
import matplotlib.pyplot as plt


from data_reader import get_data

        
if args.train:
    dev=qml.device('default.qubit',wires=args.wires+args.trash_qubits+1,shots=args.shots)
    two_comb_wires=list(combinations([i for i in range(args.wires)],2))   
    all_wires=[_ for _ in range(args.wires+args.trash_qubits+1)]
    ancillary_wires=all_wires[-1:]
    auto_wires=all_wires[:args.wires]
    ref_wires=all_wires[args.wires:args.wires+args.trash_qubits] 
elif args.test:
    test_args=Unpickle('args',path=args.path)
    dev=qml.device(args.test_device,wires=test_args.wires+test_args.trash_qubits+1,shots=test_args.shots)
    two_comb_wires=list(combinations([i for i in range(test_args.wires)],2))
    all_wires=[_ for _ in range(test_args.wires+test_args.trash_qubits+1)]
    ancillary_wires=all_wires[-1:]
    auto_wires=all_wires[:test_args.wires]
    ref_wires=all_wires[test_args.wires:test_args.wires+test_args.trash_qubits]
    del test_args
else:
    print ('Select either: --train or --test')
    sys.exit()
    
    
    
if not args.test: 
    print ('Selected argument namespace: ')
    for key,item in vars(args).items():
        print (f'\t {key:12} : {repr(item):50}')
    print ('\n')
class AutoEncData(Dataset):
    def __init__(self,data_dict,x_key='X',y_key='y',backend='autograd',double=False,size=None,
                 autoencoder=True,trash_states=-1,state_value=np.pi/2.,loc=0,std=1):
        self.x=data_dict[x_key][:size]
        if double:
            self.x=np.concatenate([self.x,self.x.copy()],axis=1)
        self.y=data_dict[y_key][:size]
        self.autoencoder=autoencoder
        if backend =='torch':
            self.x=torch.tensor(self.x.astype(np.float32))
            self.y=torch.tensor(self.y.astype(np.float32))
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.autoencoder: return self.x[idx]
        else:
            return self.x[idx],np.array([self.y[idx]])


def batch_semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        cost,fid=[],[]
        for item in inputs:
            fid.append(quantum_cost(weights,item))
            cost.append(-fid[-1])
        return np.array(cost).mean(),np.array(fid).mean()
    return np.array([-quantum_cost(weights,item) for item in inputs]).mean()


def collate_fn(samples):
    if type(samples[0])==tuple:
        X,y=[],[]
        for item in samples: X.append(item[0]),y.append(item[1])
        X,y=np.array(X),np.array(y).flatten()
        return X,y
    else: 
        return np.array(samples)
class Trainer(object):
    def __init__(self,model,lr=0.001,backend=None,
                  quantum=False,no_entangling=False,**kwargs):
        self.model=model
        self.backend=backend
        self.quantum=quantum
        self.init_weights=kwargs.get('init_weights')
        self.current_weights=self.init_weights
        self.metric_tensor=None
        self.optim=qml.QNGOptimizer(lr,diag_approx=True,lam=10e-12)
        self.quantum_loss=batch_semi_classical_cost
        self.setup_quantum()
        print (f'Performing quantum gradient with: {self.optim}  Learning rate: {lr}')
        print ('Backend:',self.backend,'\n')
        
    def setup_quantum(self):
        print ('Setting up quantum training procedure!')
        obs=[qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires])]
        coeffs=[1.]
        
        
        self.hamiltonian=qml.Hamiltonian(coeffs,obs)
        self.quantum_cost=qml.ExpvalCost(self.model,self.hamiltonian,dev,optimize=True)
        
        self.metric_tensor=qml.metric_tensor(self.quantum_cost)
        n=self.init_weights.flatten().shape[0]
        diag_reg=np.zeros((n,n))
        np.fill_diagonal(diag_reg,self.optim.lam)
        self.diag_reg=diag_reg
    def iteration(self,data,train=False,verbose=False):
        if train:
            grad,cost=self.optim.compute_grad(self.quantum_loss,[self.current_weights],dict(inputs=data,quantum_cost=self.quantum_cost))
            self.optim.metric_tensor=np.mean([self.metric_tensor(self.current_weights,inputs=item) for item in data],axis=0)+self.diag_reg
            self.current_weights=self.optim.apply_grad(grad[0],self.current_weights)
            return float(cost)
        else: 
            cost,fid=self.quantum_loss(self.current_weights,inputs=data,quantum_cost=self.quantum_cost,return_fid=True)
            return float(cost),float(fid)

    def print_params(self,prefix=None):
        if prefix is not None: print (prefix)
        print ('Autograd weights:',self.current_weights,'\n')
        
    def save(self, epoch, save_dir):
        if epoch>100: name = 'ep{:03}.pickle'.format(epoch)
        else: name='ep{:02}.pickle'.format(epoch)
        Pickle({'weights':self.current_weights},name,path=save_dir)
            

def circuit(weights,inputs=None,wires=None):

    # State preparation for all wires
    AngleEmbedding(inputs[:], wires = auto_wires, rotation="X")
    
    # QAE Circuit
    for item,i in zip(weights,auto_wires):
        qml.RY(item/2.,wires=[i])
        
    for item in two_comb_wires: 
        qml.CNOT(wires=item)
    
    # SWAP test to measure fidelity
    qml.Hadamard(ancillary_wires)
    for ref_wire,trash_wire in zip(ref_wires,auto_wires[-args.trash_qubits:]):
        qml.CSWAP(wires=[ancillary_wires[0], ref_wire, trash_wire])
    qml.Hadamard(ancillary_wires)
    
    
    
@qml.qnode(dev,interface=args.backend)
def autoenc(weights,inputs=None):

    circuit(weights,inputs=inputs)
    
    
    # returns the expected fidelity between the reference state and the trash state
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))
if args.train:
    
    

    init_weights=np.random.uniform(0,np.pi,size=(len(auto_wires),))
    
    if args.save:
        Pickle(args,'args',path=args.save_dir)
        with open(os.path.join(args.save_dir,'args.txt'),'w+') as f:
            f.write(repr(args))
    all_data=get_data(train=True,train_file=args.train_file,return_splitted=True,scale=True,bg_only=True,keys=args.keys)
    train=AutoEncData(all_data.get('train'),size=args.train_size,trash_states=0,autoencoder=True,
                          backend=args.backend)
    val=AutoEncData(all_data.get('val'),trash_states=0,size=10000,autoencoder=True,
                        backend=args.backend)
    
    
    
    trainer=Trainer(circuit,lr=args.lr,backend=args.backend,init_weights=init_weights)
    trainer.print_params('Initialized parameters!')
    train_loader=DataLoader(train,shuffle=True,batch_size=args.batch_size,collate_fn=collate_fn)
    val_loader=DataLoader(val,shuffle=False,batch_size=args.batch_size,collate_fn=collate_fn)
    
    history={'train':[],'val':[],'accuracy':[]}
    first_draw=True
    try:
        for n_epoch in range(args.epochs+1):
            if n_epoch !=0:
                losses=0.
                for data in tqdm(train_loader):
                    #print (data)
                    loss=trainer.iteration(data,train=True)
                    losses+=loss
                train_loss=losses/len(train_loader)  
            else: train_loss=float('NaN')
            print ('Validating!')
            val_loss=0.
            for data in tqdm(val_loader):
                loss,batch_fid=trainer.iteration(data,train=False)
                val_loss+=loss
            val_loss=val_loss/len(val_loader)
            print (f'Epoch {n_epoch}: Train Loss:{train_loss} Val loss: {val_loss}')
            history['train'].append(train_loss)
            history['val'].append(val_loss)
            if args.save:
                trainer.save(n_epoch,args.save_dir)
    except KeyboardInterrupt:
        pass
    finally:
        trainer.print_params('Trained parameters:')
        print (history)
        if args.save:
            done_epochs=len(history['train'])
            Pickle(history,'history',path=args.save_dir)
            fig,axes=plt.subplots(figsize=(15,12))
            axes.plot(np.arange(done_epochs),history['train'],label='train',linewidth=2)
            axes.plot(np.arange(done_epochs),history['val'],label='val',linewidth=2)
            if len(history['accuracy'])==done_epochs+1:
                axes.plot(np.arange(done_epochs),history['accuracy'],label='val accuracy',linewidth=2)
            axes.set_xlabel('Epochs',size=25)
            axes.set_ylabel('Loss/Acc',size=25)
            axes.set_xticks(np.arange(0,done_epochs,5))
            axes.legend(prop={'size':25})
    
            axes.tick_params(labelsize=20)
            fig.savefig(os.path.join(args.save_dir,'history'))
    sys.exit()

if args.test:
    curr_args=args
    print ('Testing with model:',args.path)
    history=Unpickle('history',path=args.path)
    #print (history)
    val_loss=history['val']
    epoch=np.argmin(val_loss)
    print (epoch,val_loss[epoch])
    if epoch<10:
        filename='0'+str(epoch)
    else: filename=str(epoch)
    model_path=os.path.join(args.path,'ep'+filename+'.pickle')
    print ('Model path:',model_path)
    path=args.path
    args=Unpickle('args',path=args.path)
    print ('Loaded args:',args)

    dictionary=Unpickle(model_path)
    weights=dictionary['weights']
    
    import numpy as nnp
        #init_weights=weights
    all_data=get_data(train=False,combined_signal=False,keys=args.keys,
                      train_file=args.train_file,return_splitted=True,scale=True,bg_only=True)
    

    store_dict={}
    i=0
        
    for key,val in all_data.items():
        if type(val)!= np.ndarray:
            store_dict[key]=val
            continue
       
        
    
        fid=[]
        y=[]
        print ('Testing: ',key,val.shape)
        for item in tqdm(val):
            fid.append(autoenc(weights,item))
            
        fid=nnp.array(fid)
        print (fid.shape,type(fid))
        store_dict[key]=fid
        i+=1
    print_events(store_dict,name='store dict')
    
    Pickle(store_dict,'test' if curr_args.test_device=='default.qubit' else 'test_'+'_'.join(curr_args.test_device.split('.')),path=path)
    sys.exit()
        



