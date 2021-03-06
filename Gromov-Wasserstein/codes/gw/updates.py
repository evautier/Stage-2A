
import numpy as np


def update_square_loss(p,lambdas,T,Cs):
    """
    Updates C according to the L2 Loss kernel with the S Ts couplings calculated at each iteration
    
    
    Parameters
    ----------
    p  : np.ndarray(N,)
         weights in the targeted barycenter
    lambdas : list of the S spaces' weights
    T : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Cs : Cs : list of S np.ndarray(ns,ns)
         Metric cost matrices
         
    Returns 
    ----------
    C updated
    
    
    """
    somme=sum([lambdas[s]*np.dot(T[s].T,Cs[s]).dot(T[s]) for s in range(len(T))])
    ppt=np.dot(p,p.T)
    
    return(np.divide(somme,ppt))

def update_kl_loss(p,lambdas,T,Cs):
    """
    Updates C according to the KL Loss kernel with the S Ts couplings calculated at each iteration
    
    
    Parameters
    ----------
    p  : np.ndarray(N,)
         weights in the targeted barycenter
    lambdas : list of the S spaces' weights
    T : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Cs : Cs : list of S np.ndarray(ns,ns)
         Metric cost matrices
         
    Returns 
    ----------
    C updated
    
    
    """
    somme=sum([lambdas[s]*np.dot(T[s].T,Cs[s]).dot(T[s]) for s in range(len(T))])
    ppt=np.dot(p,p.T)
    
    return(np.exp(np.divide(somme,ppt)))   