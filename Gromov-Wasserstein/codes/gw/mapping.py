# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:40:18 2017

@author: Erwan
"""

import numpy as np


def barycentric_mapping(xs,xt,gw):
    xsInt=np.zeros((xs.shape[0],xt.shape[1]))
    for i in range(xs.shape[0]):
        for d in range(xt.shape[1]):
            xsInt[i][d]=np.sum(np.multiply(gw[i],xt[:,d]))/np.sum(gw[i])
    return xsInt
    
    