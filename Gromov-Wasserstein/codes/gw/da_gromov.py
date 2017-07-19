# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:51:10 2017

@author: Erwan
"""

import numpy as np
from gromov import gromov_wasserstein
from .utils import unif,dist,kernel
import loss

def vectorizeLabels(y):
    vals=np.unique(y)
    Y=np.zeros((len(y),len(vals)))
    for i,val in enumerate(vals):
        Y[:,i]=(y==val)
    return Y

class OTDA_gromov(object):
    """Class for domain adaptation with optimal transport between two metric spaces"""
    
    def __init__(self,metric='sqeuclidean',loss=loss.tensor_square_loss):
        """ Class initialization"""
        self.xs=0
        self.xt=0
        self.G=0
        self.metric=metric
        self.L=loss
        self.computed=False
        
    def fit(self,xs,xt,reg=1,ws=None,wt=None,norm=None,**kwargs):
        """ Fit domain adaptation between samples is xs and xt (with optional weights)"""
        self.xs=xs
        self.xt=xt

        if wt is None:
            wt=unif(xt.shape[0])
        if ws is None:
            ws=unif(xs.shape[0])

        self.ws=ws
        self.wt=wt

        self.C1=dist(xs,xs,metric=self.metric)
        self.C2=dist(xt,xt,metric=self.metric)
        self.normalizeC(norm)
        self.G=gromov_wasserstein(self.C1,self.C2,ws,wt,self.L,reg,**kwargs)
        self.computed=True
    
    def predict(self,ys):
        """ Compute labels in target samples by label spreading """
        if self.computed:
            ysVec=vectorizeLabels(ys)
            ysp=len(self.xt)*self.G.T.dot(ysVec)
            yspdec=np.argmax(ysp,1)+1
            return(yspdec)
        else:
            print("Warning, model not fitted yet, returning None")
            return None
        
    def normalizeC(self, norm):
        """
        It may help to normalize the cost matrices self.C1 and self.C2 if there are numerical
        errors during the sinkhorn based algorithms.
        """
        if norm == "median":
            self.C1 /= float(np.median(self.C1))
        elif norm == "max":
            self.C1 /= float(np.max(self.C1))
        elif norm == "log":
            self.C1 = np.log(1 + self.C1)
        elif norm == "loglog":
            self.C1 = np.log(1 + np.log(1 + self.C1))
        if norm == "median":
            self.C2 /= float(np.median(self.C2))
        elif norm == "max":
            self.C2 /= float(np.max(self.C2))
        elif norm == "log":
            self.C2 = np.log(1 + self.C2)
        elif norm == "loglog":
            self.C2 = np.log(1 + np.log(1 + self.C2))