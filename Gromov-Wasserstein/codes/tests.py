# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:15:45 2017

@author: Erwan
"""

import sys
sys.path.append('C:/Users/Erwan/Desktop/Stage_2A/Gromov-Wasserstein/codes/gw')


import gromov
import loss
import mapping
import scipy as sp
import numpy as np
import ot
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

import mpl_toolkits.axes_grid1 as axes_grid1


'''
Test : 2D to 3D random
'''

n=20 # nb samples

mu_s=np.array([0,0])
cov_s=np.array([[1,0],[0,1]])

mu_t=np.array([4,4,4])
cov_t=np.array([[1,0,0],[0,1,0],[0,0,1]])



xs=ot.datasets.get_2D_samples_gauss(n,mu_s,cov_s)
P=sp.linalg.sqrtm(cov_t)
xt= np.random.randn(n,3).dot(P)+mu_t

p=ot.unif(n)
q=ot.unif(n)

fig=pl.figure()

ax1=fig.add_subplot(121)

ax1.plot(xs[:,0],xs[:,1],'+b',label='Source samples')

ax2=fig.add_subplot(122,projection='3d')

ax2.scatter(xt[:,0],xt[:,1],xt[:,2],color='r')


C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xt,xt)

C1/=C1.max()
C2/=C2.max()

pl.subplot(121)

pl.imshow(C1)

pl.subplot(122)

pl.imshow(C2)


indns=np.ones((1,n))
indnt=np.ones((1,n))

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss)

pl.imshow(gw,cmap='jet')
pl.colorbar()
pl.show()

xsInt=mapping.barycentric_mapping(xs,xt,gw)

fig=pl.figure()

ax1=fig.add_subplot(121)

ax1.plot(xs[:,0],xs[:,1],'+b',label='Source samples')

ax2=fig.add_subplot(122,projection='3d')

ax2.scatter(xt[:,0],xt[:,1],xt[:,2],color='r')
ax2.scatter(xsInt[:,0],xsInt[:,1],xsInt[:,2],color='b')

''''''''''''''''''''''''''''''

'''
Test : 3D to 2D projection
'''

n=20
mu=np.array([4,4,4])
cov=np.array([[1,0,0],[0,1,0],[0,0,1]])

P=sp.linalg.sqrtm(cov)
xs= np.random.randn(n,3).dot(P)+mu

xt=[[xs[i][0],xs[i][1]] for i in range(len(xs))]
xt=np.asarray(xs)

fig=pl.figure()

ax1=fig.add_subplot(121,projection='3d')

ax1.scatter(xs[:,0],xs[:,1],xs[:,2],color='r')

ax2=fig.add_subplot(122)

ax2.plot(xt[:,0],xt[:,1],'+b',label='Source samples')
pl.show()

p=ot.unif(n)
q=ot.unif(n)

C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xt,xt)

C1/=C1.max()
C2/=C2.max()

pl.subplot(121)

pl.imshow(C1)

pl.subplot(122)

pl.imshow(C2)

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss)

pl.imshow(gw,cmap='jet')
pl.colorbar()
pl.show()

'''''''''''''''''''''''''''''''''''
'''
'''
Test : d-D to k-D
'''

from sklearn import random_projection

n=20

d=10000

xs=np.random.randn(n,d)

transformer = random_projection.GaussianRandomProjection(eps=0.05)
xt = transformer.fit_transform(xs)
xt.shape



p=ot.unif(n)
q=ot.unif(n)

C1=sp.spatial.distance.cdist(xs,xs,'euclidean')
C2=sp.spatial.distance.cdist(xt,xt,'euclidean')

C1f=C1.reshape(1,n**2)
C1f=C1f[C1f!=0]
C1min=C1f.min()
C1max=C1f.max()

for i in range(n):
    for j in range(n):
        if C1[i][j]!=0:
            C1[i][j]=(C1[i][j]-C1min)/(C1max-C1min)
            
C2f=C2.reshape(1,n**2)
C2f=C2f[C2f!=0]
C2min=C2f.min()
C2max=C2f.max()

for i in range(n):
    for j in range(n):
        if C2[i][j]!=0:
            C2[i][j]=(C2[i][j]-C2min)/(C2max-C2min)

pl.subplot(121)

pl.imshow(C1)

pl.subplot(122)

pl.imshow(C2)

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss)

pl.imshow(gw,cmap='jet')
pl.colorbar()
pl.show()







