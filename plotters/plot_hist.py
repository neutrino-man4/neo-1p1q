import numpy as nnp
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib;matplotlib.use('Agg')
from argparse import ArgumentParser

parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--seed',default=9999,type=int,help='Some number to index the run')
parser.add_argument('--dump',default='/ceph/abal/QML/dumps',help='Dump the results to a directory')
parser.add_argument('--signals',default='grav_2p5_narrow',nargs='+',help='Signal(s) to test against')
args=parser.parse_args()

dump_dir=args.dump


qcd_fids_j1=nnp.load(os.path.join(dump_dir,'qcd_fids_j1.npy'))
qcd_fids_j2=nnp.load(os.path.join(dump_dir,'qcd_fids_j2.npy'))
qcd_costs_j1=nnp.load(os.path.join(dump_dir,'qcd_costs_j1.npy'))
qcd_costs_j2=nnp.load(os.path.join(dump_dir,'qcd_costs_j2.npy'))

fids={}
costs={}
mjj={}
labels={}
bins={}
edges={}
fpr={}
tpr={}
roc_auc={}

mjj['qcd']=nnp.load(os.path.join(dump_dir,'qcd_mjj.npy'))
fids['qcd']=nnp.maximum(qcd_fids_j1,qcd_fids_j2)
costs['qcd']=nnp.minimum(qcd_costs_j1,qcd_costs_j2)
labels['qcd']=nnp.zeros(fids['qcd'].shape[0])

bins['qcd'],edges['qcd']=nnp.histogram(fids['qcd'],density=True,bins=20,range=[90,100])
    
for signal in args.signals:
    sig_fids_j1=nnp.load(os.path.join(dump_dir,signal,'sig_fids_j1.npy'))
    sig_fids_j2=nnp.load(os.path.join(dump_dir,signal,'sig_fids_j2.npy'))
    sig_costs_j1=nnp.load(os.path.join(dump_dir,signal,'sig_costs_j1.npy'))
    sig_costs_j2=nnp.load(os.path.join(dump_dir,signal,'sig_costs_j2.npy'))
    mjj[signal]=nnp.load(os.path.join(dump_dir,signal,'sig_mjj.npy'))
    fids[signal]=nnp.maximum(sig_fids_j1,sig_fids_j2)
    costs[signal]=nnp.minimum(sig_costs_j1,sig_costs_j2)
    labels[signal]=nnp.ones(fids[signal].shape[0])
    
    bins[signal],edges[signal]=nnp.histogram(fids[signal],density=True,bins=50,range=[90,100])
    labels=nnp.concatenate([labels['qcd'],labels[signal]],axis=0)
    fids=nnp.concatenate([fids['qcd'],fids[signal]],axis=0)
    costs=nnp.concatenate([costs['qcd'],costs[signal]],axis=0) 
    fpr[signal],tpr[signal],thresholds=roc_curve(labels,costs)
    roc_auc[signal]=roc_auc_score(labels,costs)