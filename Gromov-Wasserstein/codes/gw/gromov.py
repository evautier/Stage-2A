
import sys
sys.path.append('C:/Users/Erwan/Desktop/Stage_2A/Gromov-Wasserstein/codes/gw')

import ot
import loss
import updates
import numpy as np
import scipy as sp


def gromov_wasserstein(C1,C2,p,q,loss_fun,epsilon,numItermax = 1000, stopThr=1e-9,verbose=False, log=False):
    """
    Returns the gromov-wasserstein coupling between the two measured similarity matrices
    
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
    loss_fun :  loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True 
    forcing : np.ndarray(N,2)
        list of forced couplings (where N is the number of forcing)
         
    Returns
    -------
    T : coupling between the two spaces that minimizes :
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
           
    """
    

    C1=np.asarray(C1,dtype=np.float64)
    C2=np.asarray(C2,dtype=np.float64)

    T=np.dot(p,q.T) #Initialization 
    
    cpt = 0
    err=1
    
    while (err>stopThr and cpt<numItermax):
        
        Tprev=T
        
        if loss_fun=='square_loss':
            tens=loss.tensor_square_loss(C1,C2,T)
            
        elif loss_fun=='kl_loss':
            tens=loss.tensor_kl_loss(C1,C2,T)
            
        T=ot.bregman.sinkhorn([p.T[0][i] for i in range(len(p.T[0]))],
                              [q.T[0][i] for i in range(len(q.T[0]))],tens,epsilon)
       
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            err=np.linalg.norm(T-Tprev)
            
            if log:
                log['err'].append(err)

            if verbose:
                if cpt%200 ==0:
                    print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt,err))
            
        cpt=cpt+1

    return T


def gromov_wasserstein2(C1,C2,p,q,loss_fun,epsilon,numItermax = 1000, stopThr=1e-9,verbose=False, log=False):
    """
    Returns the gromov-wasserstein discrepancy between the two measured similarity matrices
    
    (C1,p) and (C2,q)
    
    The function solves the following optimization problem:

    .. math::
        \GW_Dist = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))

    
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
    loss_fun :  loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True 
    forcing : np.ndarray(N,2)
        list of forced couplings (where N is the number of forcing)
         
    Returns
    -------
    T : coupling between the two spaces that minimizes :
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
    
    """
    
    gw=gromov_wasserstein(C1,C2,p,q,loss_fun,epsilon,numItermax,stopThr,verbose,log)

    if loss_fun=='square_loss':
        gw_dist=np.sum([loss.square_loss(C1[i][k],C2[j][l])*gw[i][j]*gw[k][l] for i in range(len(C1)) 
                                                                              for k in range(len(C1)) 
                                                                              for j in range(len(C2)) 
                                                                              for l in range(len(C2))])
    
    elif loss_fun=='kl_loss':
        gw_dist=np.sum([loss.kl_loss(C1[i][k],C2[j][l])*gw[i][j]*gw[k][l] for i in range(len(C1)) 
                                                                          for k in range(len(C1)) 
                                                                          for j in range(len(C2)) 
                                                                          for l in range(len(C2))])
  
    return gw_dist


def gromov_barycenters(N,Cs,ps,p,lambdas,loss_fun,epsilon,numItermax = 1000, stopThr=1e-9, verbose=False, log=False):
    """
    Returns the gromov-wasserstein barycenters of S measured similarity matrices
    
    (Cs)_{s=1}^{s=S}
    
    The function solves the following optimization problem:

    .. math::
        C = argmin_C\in R^NxN \sum_s \lambda_s GW(C,Cs,p,ps)

    
    Where :
        
        Cs : metric cost matrix
        ps  : distribution
            
    Parameters
    ----------
    N  : Integer 
         Size of the targeted barycenter
    Cs : list of S np.ndarray(ns,ns)
         Metric cost matrices
    ps : list of S np.ndarray(ns,)
         sample weights in the S spaces
    p  : np.ndarray(N,)
         weights in the targeted barycenter
    lambdas : list of the S spaces' weights
    L :  tensor-matrix multiplication function based on specific loss function
    update : function(p,lambdas,T,Cs) that updates C according to a specific Kernel
             with the S Ts couplings calculated at each iteration
    epsilon : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True 
         
    Returns
    -------
    C : Similarity matrix in the barycenter space (permutated arbitrarily)
           
    """
    
    
    S=len(Cs)
    
    Cs=[np.asarray(Cs[s],dtype=np.float64) for s in range(S)]
    lambdas=np.asarray(lambdas,dtype=np.float64)

    T=[0 for s in range(S)]
    
    #Initialization of C : random SPD matrix
    xalea=np.random.randn(N,2)
    C=sp.spatial.distance.cdist(xalea,xalea)
    C/=C.max()
    
    cpt=0
    err=1
    
    error=[]
    
    while(err>stopThr and cpt<numItermax):
        
        Cprev=C
        
        T=[gromov_wasserstein(Cs[s],C,ps[s],p,loss_fun,epsilon,numItermax,1e-5,verbose,log) for s in range(S)]
        
        if loss_fun=='square_loss':
            C=updates.update_square_loss(p,lambdas,T,Cs)
            
        elif loss_fun=='kl_loss':
            C=updates.update_kl_loss(p,lambdas,T,Cs)
        
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            err=np.linalg.norm(C-Cprev)
            error.append(err)
            
            if log:
                log['err'].append(err)

            if verbose:
                if cpt%200 ==0:
                    print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt,err))
                
        cpt=cpt+1
        
    return C
    


    