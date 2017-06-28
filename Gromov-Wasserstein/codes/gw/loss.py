# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:16:46 2017

@author: Erwan
"""


import numpy as np


def tensor_square_loss(C1,C2,T):
    """
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the square loss 
    
    function as the loss function of Gromow-Wasserstein discrepancy.
    
    Where :
        
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        T : A coupling between those two spaces
        
    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            f1(a)=(a^2)/2
            f2(b)=(b^2)/2
            h1(a)=a
            h2(b)=b
            
    Parameters
    ----------
    C1 : np.ndarray(ns,ns)
         Metric cost matrix in the source space
    C2 : np.ndarray(nt,nt)
         Metric costfr matrix in the target space
    T :  np.ndarray(ns,nt)
         Coupling between source and target spaces


    Returns
    -------
    tens : (ns*nt) ndarray
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
           
           
    """
    

    C1=np.asarray(C1,dtype=np.float64)
    C2=np.asarray(C2,dtype=np.float64)
    T=np.asarray(T,dtype=np.float64)
    
    
    def f1(a):
        return (a**2)/2
    
    def f2(b):
        return (b**2)/2
    
    def h1(a):
        return a
    
    def h2(b):
        return b
    
    tens=-np.dot(h1(C1),T).dot(h2(C2).T)
    tens=tens-tens.min()
    
    return tens
    
    
    
    
    