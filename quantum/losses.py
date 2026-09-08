"""
Loss functions for fidelity objectives and quantum classifiers.

Author: Aritra Bal (ETP)
Date: 2026-09-08
"""

import pennylane.numpy as np
import tqdm
from typing import Callable, Dict
import quantum.math_functions as mfunc
from autograd.scipy.special import expit

def sigmoid(x):
    """Convert logits to probabilities with an elementwise stable sigmoid."""
    return expit(x)

# name -> scoring function(labels, score); add an entry here to support a new loss_type
LOSS_FNS: Dict[str, Callable] = {
    'MSE': mfunc.mean_squared_error,
    'BCE': mfunc.binary_cross_entropy_with_logits,
}

def semi_classical_cost(weights,inputs=None,quantum_circuit=None,return_fid=False):
    """Return 100 times one minus the circuit fidelity, without averaging.

    Call quantum_circuit(weights, inputs) for the fidelity. If return_fid is
    True, return (cost, 100 * fidelity).
    """
    if return_fid:
        fid=quantum_circuit(weights,inputs)
        #fid=np.sqrt(fid)
        cost=1.-fid
        return 100.*np.array(cost),100.*np.array(fid)
    
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    fid=quantum_circuit(weights,inputs)
    #fid=np.sqrt(fid)
    cost=1-fid#np.array([-quantum_circuit(weights,item) for item in inputs]).mean()
    return 100.*np.array(cost,requires_grad=False)

def batch_semi_classical_cost(weights,inputs=None,quantum_circuit=None,return_fid=False):
    """Return the mean infidelity scaled by 100 over the circuit outputs.

    If return_fid is True, return (mean cost, mean fidelity), both scaled
    by 100.
    """
    if return_fid:
        fid=quantum_circuit(weights,inputs)
        #fid=np.sqrt(fid)
        cost=1.-fid
        return np.array(100.*cost,requires_grad=True).mean(),np.array(100.*fid,requires_grad=True).mean()
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    fid=quantum_circuit(weights,inputs)
    #fid=np.sqrt(fid)
    cost=1-fid
    batched_average_cost=100.*(np.array(cost,requires_grad=True).mean())#np.array([-quantum_circuit(weights,item) for item in inputs]).mean()
    return batched_average_cost

def VQC_cost(weights,inputs=None,quantum_circuit=None,labels=None,return_scores=False,loss_type='MSE',reg=1.):
    """Compute mean classifier loss from circuit expectations plus weights.aux['bias'].

    Flatten the circuit outputs and apply the loss registered under loss_type.
    BCE consumes logits; MSE consumes raw scores. If return_scores is True,
    return (loss, scores), with sigmoid probabilities for BCE and raw scores
    otherwise. Unknown loss names raise ValueError. reg is unused.
    """
    bias=weights.aux['bias']
    exp_vals=np.reshape(
        np.array(quantum_circuit(weights,inputs),requires_grad=True),
        (-1,),
    )
    score=exp_vals+bias
    if loss_type not in LOSS_FNS:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Registered: {list(LOSS_FNS)}")
    loss_fn=LOSS_FNS[loss_type](labels,score)
    if return_scores:
        return loss_fn, sigmoid(score) if loss_type == 'BCE' else score
    return np.array(loss_fn,requires_grad=True)

def batched_VQC_cost(weights,inputs=None,quantum_circuit=None,labels=None,return_scores=False,loss_type='MSE',reg=1.):
    """Evaluate inputs individually and return the mean registered classifier loss.

    Add weights.aux['bias'] to each circuit output before computing the loss.
    If return_scores is True, return arrays of individual losses and scores
    instead of averaging. BCE scores are sigmoid probabilities; other scores
    are raw outputs plus bias. Unknown loss names raise ValueError. reg is unused.
    """
    bias=weights.aux['bias']
    #k1=weights[-3]
    #k2=weights[-3]
    if loss_type not in LOSS_FNS:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Registered: {list(LOSS_FNS)}")
    loss_fn=[]
    scores=[]
    for input,label in tqdm.tqdm(zip(inputs,labels),total=len(inputs)):
        exp_vals=np.array(quantum_circuit(weights,input[None,...]),requires_grad=True) # n_qubits x batch_size
        #exp_vals=np.mean(exp_vals,axis=0)
        score=exp_vals+bias
        scores.append(sigmoid(score) if loss_type == 'BCE' else score)
        loss_fn.append(LOSS_FNS[loss_type](label,score))
    loss_fn=np.array(loss_fn,requires_grad=False)
    score=np.array(scores,requires_grad=False)
    if return_scores:
        return loss_fn, score
    return np.mean(loss_fn)

def probabilistic_loss(weights,inputs=None,quantum_circuit=None,labels=None,return_scores=False,loss_type='BCE'):
    """Return the mean of one minus each labeled class probability.

    Expect a class-probability vector for one sample or a (batch, classes)
    array for multiple samples, with integer class labels. Scalar VQC
    expectation outputs are incompatible. If return_scores is True, return
    (mean loss, individual probability deficits). loss_type is unused.
    """
    batch_size=len(labels)
    probs=np.array(quantum_circuit(weights,inputs),requires_grad=True) # n_qubits x batch_size
    
    if batch_size==1:
        scores=1-probs[labels[0]]
    else:
        scores=1-probs[np.arange(probs.shape[0]), labels]
    loss_fn=np.mean(scores)
        
    if return_scores:
        return loss_fn, scores
    return np.array(loss_fn,requires_grad=True)
