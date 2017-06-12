# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import ot
import numpy as np

def gromov_wasserstein(C1,C2,p,q,L,numItermax = 1000, stopThr=1e-9, verbose=False, log=False):
    """
    Returns the gromov-wasserstein discrepancy between the two measured similarity matrices
    
    (C1,p) and (C2,q)
    
    The function solves the following optimization problem:

    .. math::
        \GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))

        s.t. \GW 1 = p

             \GW^T 1= q

             \GW\geq 0
    
    Where :
        
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        p  : distribution in the source space
        q  : distribution in the target space 
        L  : loss function to account for the misfit between the similarity matrices
        H  : entropy
        
            
    Parameters
    ----------
    C1 : np.ndarray(ns,ns)
         Metric cost matrix in the source space
    C2 : np.ndarray(nt,nt)
         Metric costfr matrix in the target space
    p :  np.ndarray(ns,)
         distribution in the source space
    q :  np.ndarray(nt)
         distribution in the target space
    L :  tensor-matrix multiplication function based on specific loss function
         
    Returns
    -------
    
           
           
    """
    
    p=np.asarray(p,dtype=np.float64)
    q=np.asarray(q,dtype=np.float64)
    C1=np.asarray(C1,dtype=np.float64)
    C2=np.asarray(C2,dtype=np.float64)

    ns=len(p)
    nt=len(q)
    
    indns=np.ones((1,ns))
    indnt=np.ones((1,nt))
    
    T=np.ones(ns*nt)/(ns*nt)
    T=np.reshape(T,(ns,nt))
    
    
    cpt = 0
    err=1
    
    while (err>stopThr and cpt<numItermax):
        Tprev=T
        tens=L(C1,C2,T,p,q)
        gw=ot.bergman.sinkhorn(p,q,tens,0.1)
        if (np.any(T==0)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            T=Tprev
            break
        cpt=cpt+1
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            err=
    
    
    return gw
    
    