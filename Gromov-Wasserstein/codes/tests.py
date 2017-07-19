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
import time
import scipy as sp
import numpy as np
import updates
import ot
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
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

gw=gromov.gromov_wasserstein(C1,C2,p,q,'square_loss',epsilon=5e-4)

gw_dist=gromov.gromov_wasserstein2(C1,C2,p,q,'square_loss',epsilon=5e-4)

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

def time_n(n):
    mu_s=np.array([0,0])
    cov_s=np.array([[1,0],[0,1]])
    mu_t=np.array([4,4,4])
    cov_t=np.array([[1,0,0],[0,1,0],[0,0,1]])
    xs=ot.datasets.get_2D_samples_gauss(n,mu_s,cov_s)
    P=sp.linalg.sqrtm(cov_t)
    xt= np.random.randn(n,3).dot(P)+mu_t
    p=ot.unif(n)
    q=ot.unif(n)
    C1=sp.spatial.distance.cdist(xs,xs)
    C2=sp.spatial.distance.cdist(xt,xt)
    C1/=C1.max()
    C2/=C2.max()
    tps1=time.clock()
    gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss)
    tps2=time.clock()
    return(tps2-tps1)

n=np.linspace(0,1000,21)
times=np.zeros(21)
for i in range(len(n)):
    times=time_n(n[i])

        

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
xt=np.asarray(xt)

xs=xt

fig=pl.figure()

ax1=fig.add_subplot(121,projection='3d')

ax1.scatter(xs[:,0],xs[:,1],xs[:,2],color='r')

ax2=fig.add_subplot(122)

ax2.plot(xt[:,0],xt[:,1],'+b',label='Source samples')
pl.show()

p=ot.unif(n)
q=ot.unif(n)
p=p.reshape(1,len(p)).T
q=q.reshape(1,len(q)).T

C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xt,xt)

C1/=C1.max()
C2/=C2.max()

pl.subplot(121)

pl.imshow(C1)

pl.subplot(122)

pl.imshow(C2)

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4)

pl.imshow(gw,cmap='jet')
pl.colorbar()
pl.show()


lambdas=[0,1]
Cs=[C1,C2]
ps=[p,q]

Cbary=gromov.gromov_barycenters(20,Cs,ps,p,lambdas,loss.tensor_square_loss,updates.update_square_loss,5e-4)

nposS=mapping.smacof_mds(Cbary,2)
nposT=mapping.smacof_mds(C2,2)

clf = PCA(n_components=2)

nposS=clf.fit_transform(nposS)
nposT=clf.fit_transform(nposT)

fig=pl.figure()

ax1=fig.add_subplot(121,projection='3d')

ax1.scatter(xs[:,0],xs[:,1],xs[:,2],color='b')


fig=pl.figure()
ax2=fig.add_subplot(122)
ax2.scatter(nposS[:,0],nposS[:,1],color='r')
ax2.scatter(nposT[:,0],nposT[:,1],color='b',marker='+')
pl.show()


import imp
imp.reload(loss)
imp.reload(gromov)
imp.reload(updates)
imp.reload(mapping)

'''''''''''''''''''''''''''''''''''
'''
'''
Test : d-D to k-D
'''

from sklearn import random_projection

n=20

d=10000

xs=np.random.randn(n,d)

transformer = random_projection.GaussianRandomProjection(eps=0.1)
xt = transformer.fit_transform(xs)
xt.shape



p=ot.unif(n)
q=ot.unif(n)

p=p.reshape(1,len(p))
q=q.reshape(1,len(q))

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


'''
Test 3D to 2Dprojection : separated samples
'''

n=20
mu1=np.array([8,8,8])
cov1=np.array([[1,0,0],[0,1,0],[0,0,1]])

mu2=np.array([1,1,1])
cov2=np.array([[1,0,0],[0,1,0],[0,0,1]])

P1=sp.linalg.sqrtm(cov1)
xs1= np.random.randn(10,3).dot(P1)+mu1
P2=sp.linalg.sqrtm(cov2)
xs2= np.random.randn(10,3).dot(P2)+mu2
xs=np.concatenate((xs1,xs2))

xt=[[xs[i][0],xs[i][1]] for i in range(len(xs))]
xt=np.asarray(xt)

fig=pl.figure()

ax1=fig.add_subplot(121,projection='3d')

ax1.scatter(xs[:,0],xs[:,1],xs[:,2],color='r')

ax2=fig.add_subplot(122)

ax2.plot(xt[:,0],xt[:,1],'+b',label='Source samples')
pl.show()

p=ot.unif(n)
q=ot.unif(n)
p=p.reshape(1,len(p))
q=q.reshape(1,len(q))

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

xsInt=mapping.barycentric_mapping(xs,xt,gw)

fig=pl.figure()

ax1=fig.add_subplot(121,projection='3d')

ax1.scatter(xs[:,0],xs[:,1],xs[:,2],color='b')

ax2=fig.add_subplot(122)

ax2.plot(xt[:,0],xt[:,1],'+b',label='Source samples')
ax2.scatter(xsInt[:,0],xsInt[:,1],color='r')



'''
Test : 10D to 3D projection
'''

n=20
mu=np.zeros(10)
cov=np.eye(10)

P=sp.linalg.sqrtm(cov)
xs= np.random.randn(n,10).dot(P)+mu

xt=[[xs[i][0],xs[i][2],xs[i][7]] for i in range(len(xs))]
xt=np.asarray(xt)

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

xsInt=mapping.barycentric_mapping(xs,xt,gw)

fig=pl.figure()

ax1=fig.add_subplot(111,projection='3d')

ax1.plot(xt[:,0],xt[:,1],xt[:,2],'+b',label='Source samples')
ax1.scatter(xsInt[:,0],xsInt[:,1],xsInt[:,2],color='r')


'''
TEST
'''


n=20
mu=np.array([4,4,4])
cov=np.array([[1,0,0],[0,1,0],[0,0,1]])

P=sp.linalg.sqrtm(cov)
xs= np.random.randn(n,3).dot(P)+mu

xt=[[xs[i][0],xs[i][1]] for i in range(len(xs))]
xt=np.asarray(xt)

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

xsInt=mapping.barycentric_mapping(xs,xt,gw)

fig=pl.figure()

ax1=fig.add_subplot(121,projection='3d')

ax1.scatter(xs[:,0],xs[:,1],xs[:,2],color='b')

ax2=fig.add_subplot(122)

ax2.plot(xt[:,0],xt[:,1],'+b',label='Source samples')
ax2.scatter(xsInt[:,0],xsInt[:,1],color='r')

def mapping(xs,xt,gw):
    xsInt2=np.zeros((len(xs),2))
    xsInt2=np.asarray(xsInt2)
    for i in range(len(xs)):
        for e in range(1000):
            x=np.random.randn(1,2)+np.array([4,4])
            if (f(x,xs,xt,gw,xs[i]) < f(xsInt2[i],xs,xt,gw,xs[i])):
                xsInt2[i]=x
    return xsInt2



def f(x,xs,xt,gw,i):
    res=0
    x=x.reshape((1,2))
    for k in range(len(xs)):
        dik=((i[0]-xs[k][0])**2+(i[1]-xs[k][1])**2+(i[2]-xs[k][2])**2)
        for l in range(len(xt)):
            dxl=((x[0][0]-xt[l][0])**2+(x[0][1]-xt[l][1])**2)
            res=res+(1/2)*(dik-dxl)**2*gw[k][l]
    return res


def test(xs,xt,gw,i,nb):
    mini=10000
    for e in range(nb):
            x=np.random.randn(1,2)+np.array([4,4])
            fi=f(x,xs,xt,gw,xs[i])
            if (fi < mini):
                mini=fi
    return mini

test=[[nb,test(xs,xt,gw,4,nb)] for nb in [10,100,500,1000,5000,10000,50000]]

xsInt2=mapping(xs,xt,gw)


fig=pl.figure()

ax1=fig.add_subplot(121,projection='3d')

ax1.scatter(xs[:,0],xs[:,1],xs[:,2],color='b')

ax2=fig.add_subplot(122)

ax2.plot(xt[:,0],xt[:,1],'+b',label='Source samples')
ax2.scatter(xsInt2[:,0],xsInt2[:,1],color='r')


'''
Test Gromow-Wasserstein Barycenters
'''

S=3
ns=[20,26,31]

xs=[0 for s in range(S)]

xs[0]=np.random.randn(ns[0],2)

mu1=np.array([8,4])
cov1=np.eye(2)
P1=sp.linalg.sqrtm(cov1)

mu2=np.array([0,8])
cov2=np.eye(2)
P2=sp.linalg.sqrtm(cov2)

xs[1]=np.random.randn(ns[1],2).dot(P1)+mu1
xs[2]=np.random.randn(ns[2],2).dot(P2)+mu2


Cs=[sp.spatial.distance.cdist(xs[s],xs[s]) for s in range(S)]
Cs=[cs/cs.max() for cs in Cs]

fig=pl.figure()

pl.scatter(xs[0][:,0],xs[0][:,1],color='r')

pl.scatter(xs[1][:,0],xs[1][:,1],color='b')

pl.scatter(xs[2][:,0],xs[2][:,1],color='g')

pl.show()


ps=[ot.unif(ns[s]) for s in range(S)]
p=ot.unif(30)
p=p.reshape(1,len(p))
ps=[ps[s].reshape(1,len(ps[s])) for s in range(S)]

lambdas=[1/S for s in range(S)]

C=gromov.gromov_barycenters(30,Cs,ps,p,lambdas,loss.tensor_square_loss,updates.update_square_loss,numItermax=1000)

xalea=np.random.randn(30,2)
C=sp.spatial.distance.cdist(xalea,xalea)
C/=C.max()
npos=mapping.smacof_mds(C,2)
npos *= np.sqrt((xalea ** 2).sum()) / np.sqrt((npos ** 2).sum())

clf = PCA(n_components=2)
X_true = clf.fit_transform(xalea)

pos = clf.fit_transform(pos)

npos = clf.fit_transform(npos)

pl.scatter(xalea[:,0],xalea[:,1],color='r')

pl.scatter(npos[:,0],npos[:,1],color='b')


pl.scatter(xs[0][:,0],xs[0][:,1],color='r')

pl.scatter(xs[1][:,0],xs[1][:,1],color='b')

pl.scatter(xs[2][:,0],xs[2][:,1],color='g')

pl.scatter(npos[:,0],npos[:,1],color='black',marker='+')

xalea=np.random.randn(30,2)
C=sp.spatial.distance.cdist(xalea,xalea)
C/=C.max()

T=[0,0,0]

T[0]=gromov.gromov_wasserstein(Cs[0],C,ps[0],p,loss.tensor_square_loss)
T[1]=gromov.gromov_wasserstein(Cs[1],C,ps[1],p,loss.tensor_square_loss)
T[2]=gromov.gromov_wasserstein(Cs[2],C,ps[2],p,loss.tensor_square_loss)


list=[lambdas[s]*np.dot(T[s].T,Cs[s]).dot(T[s]) for s in range(len(T))]
somme=sum(list)
ppt=np.dot(p,p.T)
C=np.divide(somme,ppt)


import imp
imp.reload(loss)
imp.reload(gromov)
imp.reload(updates)
imp.reload(mapping)



'''
TEST : same spaces
'''


S=3
ns=[30,30,30]
N=30

xs=[0 for s in range(S)]

xs[0]=np.random.randn(ns[0],2)
xs[1]=xs[0]
xs[2]=xs[0]

Cs=[sp.spatial.distance.cdist(xs[s],xs[s]) for s in range(S)]
Cs=[cs/cs.max() for cs in Cs]

ps=[ot.unif(ns[s]) for s in range(S)]
p=ot.unif(N)
p=p.reshape(1,len(p)).T
ps=[ps[s].reshape(1,len(ps[s])).T for s in range(S)]


lambdas=[1/S for s in range(S)]

C=gromov.gromov_barycenters(N,Cs,ps,p,lambdas,loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=1000,stopThr=1e-3)

def donnees_bary(epsilon):
    C=gromov.gromov_barycenters(N,Cs,ps,p,lambdas,loss.tensor_square_loss,updates.update_square_loss,epsilon,numItermax=1000,stopThr=1e-3)
    nposbary=mapping.smacof_mds(C,2)
    nposbary=clf.fit_transform(nposbary)
    return(nposbary)



xalea=np.random.randn(30,2) 
C=sp.spatial.distance.cdist(xalea,xalea)
C/=C.max()

T=[0,0,0]
T[0]=gromov.gromov_wasserstein(Cs[0],C,ps[0],p,loss.tensor_square_loss)
T[1]=gromov.gromov_wasserstein(Cs[1],C,ps[1],p,loss.tensor_square_loss)
T[2]=gromov.gromov_wasserstein(Cs[2],C,ps[2],p,loss.tensor_square_loss)

list=[lambdas[s]*np.dot(T[s].T,Cs[s]).dot(T[s]) for s in range(len(T))]
somme=sum(list)
ppt=np.dot(p,p.T)
C=np.divide(somme,ppt)



T[0]=gromov.gromov_wasserstein(Cs[0],C,ps[0],p,loss.tensor_square_loss)
T[1]=gromov.gromov_wasserstein(C,Cs[0],p,ps[0],loss.tensor_square_loss)


Csums=[]
C0sums=[]
for i in range(len(C)):
    Csums.append(C[i].sum())
    C0sums.append(Cs[0][i].sum())



donnees=np.zeros(5)
donnees=[donnees_bary(epsilon) for epsilon in [5e-1,5e-2,5e-3,5e-5,5e-6]]



npos=[0,0,0]
npos=[mapping.smacof_mds(Cs[s],2) for s in range(S)]

nposbary=mapping.smacof_mds(C,2)

clf = PCA(n_components=2)

npos=[clf.fit_transform(npos[s]) for s in range(S)]
nposbary=clf.fit_transform(nposbary)


pl.scatter(npos[0][:,0],npos[0][:,1],color='b')
pl.scatter(nposbary[:,0],nposbary[:,1],color='r',marker='+')



pl.scatter(npos[1][:,0],npos[1][:,1],color='b',s=100)

pl.scatter(npos[2][:,0],npos[2][:,1],color='g',s=20)



'''
TEST : CIRCLES
'''

def rotation(x,angle):
    xnew=x
    for i in range(len(x)):
        xnew[i][0]=x[i][0]*np.cos(angle)-x[i][1]*np.sin(angle)
        xnew[i][1]=x[i][0]*np.sin(angle)+x[i][1]*np.cos(angle)
    return xnew


S=3
ns=[100,110,90]
N=100


xs=[0 for s in range(S)]

theta0 = 2*np.pi*np.random.uniform(0,1,(ns[0],1))
r01 = 8 + 2*np.random.uniform(0,1,(ns[0],1))
r02 = 6 + 2*np.random.uniform(0,1,(ns[0],1))
x00=np.multiply(np.cos(theta0),r01)
x01=np.multiply(np.sin(theta0),r02)
xs[0] =np.concatenate((x00,x01),axis=1)
xs[0]=1.6*rotation(xs[0],90)

theta1 = 2*np.pi*np.random.uniform(4,6,(ns[1],1))
r11 = 6 + 2*np.random.uniform(0,1,(ns[1],1))
r12 = 19 + 2*np.random.uniform(0,1,(ns[1],1))
x10=np.multiply(np.cos(theta1),r11)
x11=np.multiply(np.sin(theta1),r12)
xs[1] =np.concatenate((x10,x11),axis=1)
xs[1]=1.5*rotation(xs[1],60)

theta2 = 2*np.pi*np.random.uniform(2,7,(ns[2],1))
r21 = 2 + 2*np.random.uniform(0,1,(ns[2],1))
r22 = 28 + 2*np.random.uniform(0,1,(ns[2],1))
x20=np.multiply(np.cos(theta2),r21)
x21=np.multiply(np.sin(theta2),r22)
xs[2] =np.concatenate((x20,x21),axis=1)
xs[2]=1.9*rotation(xs[2],120)


pl.scatter(xs[0][:,0],xs[0][:,1],s=10)
pl.scatter(xs[1][:,0],xs[1][:,1],s=10)
pl.scatter(xs[2][:,0],xs[2][:,1],s=10)

Cs=[sp.spatial.distance.cdist(xs[s],xs[s]) for s in range(S)]
Cs=[cs/cs.max() for cs in Cs]

ps=[ot.unif(ns[s]) for s in range(S)]
p=ot.unif(N)
p=p.reshape(1,len(p)).T
ps=[ps[s].reshape(1,len(ps[s])).T for s in range(S)]


lambdas=[1/S for s in range(S)]


fig=pl.figure()

ax1=fig.add_subplot(131)
ax1.imshow(Cs[0])

ax2=fig.add_subplot(132)
ax2.imshow(Cs[1])

ax3=fig.add_subplot(133)
ax3.imshow(Cs[2])

C=gromov.gromov_barycenters(N,Cs,ps,p,lambdas,loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=1000,stopThr=1e-3)


npos=[0,0,0]
npos=[mapping.smacof_mds(Cs[s],2) for s in range(S)]

nposbary=mapping.smacof_mds(C,2)

clf = PCA(n_components=2)

npos=[clf.fit_transform(npos[s]) for s in range(S)]
nposbary=clf.fit_transform(nposbary)



pl.scatter(npos[0][:,0],npos[0][:,1],color='b',s=5)
pl.scatter(npos[1][:,0],npos[1][:,1],color='orange',s=5)
pl.scatter(npos[2][:,0],npos[2][:,1],color='g',s=5)
pl.scatter(nposbary[:,0],nposbary[:,1],color='r',marker='+')


fig=pl.figure()

ax1=fig.add_subplot(221)
ax1.scatter(npos[0][:,0],npos[0][:,1],color='b',s=5)

ax2=fig.add_subplot(222)
ax2.scatter(npos[1][:,0],npos[1][:,1],color='orange',s=5)

ax3=fig.add_subplot(223)
ax3.scatter(npos[2][:,0],npos[2][:,1],color='g',s=5)

ax4=fig.add_subplot(224)
ax4.scatter(nposbary[:,0],nposbary[:,1],color='r',marker='+')


'''
TEST : SHAPE PROGRESSIVE INTERPOLATION
'''

S=2
ns=[100,100]
N=100


xs=[0 for s in range(S)]

xs[0]=np.random.uniform(0,1,(ns[0],2))-0.5

theta = 2*np.pi*np.random.uniform(0,1,(ns[1],1))
r =np.random.uniform(0,0.5,(ns[1],1))
x10=np.multiply(np.cos(theta),r)
x11=np.multiply(np.sin(theta),r)
xs[1] =np.concatenate((x10,x11),axis=1)

pl.scatter(xs[0][:,0],xs[0][:,1])
pl.scatter(xs[1][:,0],xs[1][:,1],color='r')

Cs=[sp.spatial.distance.cdist(xs[s],xs[s]) for s in range(S)]
Cs=[cs/cs.max() for cs in Cs]

ps=[ot.unif(ns[s]) for s in range(S)]
p=ot.unif(N)
p=p.reshape(1,len(p)).T
ps=[ps[s].reshape(1,len(ps[s])).T for s in range(S)]




npos=[0,0]
npos=[mapping.smacof_mds(Cs[s],2) for s in range(S)]


clf = PCA(n_components=2)

npos=[clf.fit_transform(npos[s]) for s in range(S)]


fig=pl.figure()

ax1=fig.add_subplot(121)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax1.scatter(npos[0][:,0],npos[0][:,1],color='b')

ax2=fig.add_subplot(122)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax2.scatter(npos[1][:,0],npos[1][:,1],color='r')

Ct=[0 for i in range(5)]
lambdast=[[i/5,(5-i)/5] for i in [1,2,3,4]]

for i in range(4):
    Ct[i]=gromov.gromov_barycenters(N,Cs,ps,p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=1000,stopThr=1e-3)


npost=[0,0,0,0]
npost=[mapping.smacof_mds(Ct[s],2) for s in range(4)]

npos=[clf.fit_transform(npos[s]) for s in range(S)]


fig=pl.figure(figsize=(15,10))

ax1=fig.add_subplot(321)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax1.scatter(npos[0][:,0],npos[0][:,1],color='r')

ax2=fig.add_subplot(322)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax2.scatter(npost[3][:,0],npost[3][:,1],color='b')

ax3=fig.add_subplot(323)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax3.scatter(npost[2][:,0],npost[2][:,1],color='b')

ax4=fig.add_subplot(324)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax4.scatter(npost[1][:,0],npost[1][:,1],color='b')

ax5=fig.add_subplot(325)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax5.scatter(npost[0][:,0],npost[0][:,1],color='b')

ax6=fig.add_subplot(326)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax6.scatter(npos[1][:,0],npos[1][:,1],color='r')


'''
TEST : Gaussian distributions
'''

n=100

a1=ot.datasets.get_1D_gauss(n,m=20,s=5)
a2=ot.datasets.get_1D_gauss(n,m=60,s=8)
x=np.arange(n,dtype=np.float64)

pl.plot(x,a1)
pl.plot(x,a2)


'''
TEST : MNIST
'''

from sklearn.datasets import load_digits
import os
digits = load_digits()

pl.imshow(digits.images[0],cmap='gray')
pl.imshow(digits.images[1],cmap='gray')
pl.imshow(digits.images[2],cmap='gray')
pl.imshow(digits.images[3],cmap='gray')

zero=digits.images[0]
one=digits.images[1]
two=digits.images[2]
three=digits.images[3]

numbers=[zero,one,two,three]



S=4
xs=[[] for i in range(S)]


for nb in range(4):
    for i in range(8):
        for j in range(8):
            if numbers[nb][i,j]!=0:
                xs[nb].append([j,8-i])

xs=np.array([np.array(xs[0]),np.array(xs[1]),np.array(xs[2]),np.array(xs[3])])

ns=[len(xs[s]) for s in range(S)]
N=30

fig=pl.figure(figsize=(15,10))

ax1=fig.add_subplot(221)
pl.xlim((-1,8))
pl.ylim((0,9))
ax1.scatter(xs[0][:,0],xs[0][:,1],color='b')

ax2=fig.add_subplot(222)
pl.xlim((-1,8))
pl.ylim((0,9))
ax2.scatter(xs[1][:,0],xs[1][:,1],color='b')

ax3=fig.add_subplot(223)
pl.xlim((-1,8))
pl.ylim((0,9))
ax3.scatter(xs[2][:,0],xs[2][:,1],color='b')

ax4=fig.add_subplot(224)
pl.xlim((-1,8))
pl.ylim((0,9))
ax4.scatter(xs[3][:,0],xs[3][:,1],color='b')


Cs=[sp.spatial.distance.cdist(xs[s],xs[s]) for s in range(S)]
Cs=[cs/cs.max() for cs in Cs]

ps=[ot.unif(ns[s]) for s in range(S)]
p=ot.unif(N)
p=p.reshape(1,len(p)).T
ps=[ps[s].reshape(1,len(ps[s])).T for s in range(S)]


npos=[0,0,0,0]
npos=[mapping.smacof_mds(Cs[s],2) for s in range(S)]

clf = PCA(n_components=2)

npos=[clf.fit_transform(npos[s]) for s in range(S)]

fig=pl.figure(figsize=(15,10))

ax1=fig.add_subplot(221)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax1.scatter(npos[0][:,0],npos[0][:,1],color='b')

ax2=fig.add_subplot(222)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax2.scatter(npos[1][:,0],npos[1][:,1],color='b')

ax3=fig.add_subplot(223)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax3.scatter(npos[2][:,0],npos[2][:,1],color='b')

ax4=fig.add_subplot(224)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax4.scatter(npos[3][:,0],npos[3][:,1],color='b')



lambdast=[[i/3,(3-i)/3] for i in [1,2]]

Ct01=[0 for i in range(2)]
for i in range(2):
    Ct01[i]=gromov.gromov_barycenters(N,[Cs[0],Cs[1]],[ps[0],ps[1]],p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=1000,stopThr=1e-3)

Ct02=[0 for i in range(2)]
for i in range(2):
    Ct02[i]=gromov.gromov_barycenters(N,[Cs[0],Cs[2]],[ps[0],ps[2]],p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=1000,stopThr=1e-3)

Ct13=[0 for i in range(2)]
for i in range(2):
    Ct13[i]=gromov.gromov_barycenters(N,[Cs[1],Cs[3]],[ps[1],ps[3]],p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=1000,stopThr=1e-3)

Ct23=[0 for i in range(2)]
for i in range(2):
    Ct23[i]=gromov.gromov_barycenters(N,[Cs[2],Cs[3]],[ps[2],ps[3]],p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=1000,stopThr=1e-3)



npost01=[0,0]
npost01=[mapping.smacof_mds(Ct01[s],2) for s in range(2)]
npost01=[clf.fit_transform(npost01[s]) for s in range(2)]

npost02=[0,0]
npost02=[mapping.smacof_mds(Ct02[s],2) for s in range(2)]
npost02=[clf.fit_transform(npost02[s]) for s in range(2)]

npost13=[0,0]
npost13=[mapping.smacof_mds(Ct13[s],2) for s in range(2)]
npost13=[clf.fit_transform(npost13[s]) for s in range(2)]

npost23=[0,0]
npost23=[mapping.smacof_mds(Ct23[s],2) for s in range(2)]
npost23=[clf.fit_transform(npost23[s]) for s in range(2)]



fig=pl.figure(figsize=(10,10))

ax1=pl.subplot2grid((4,4), (0,0))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax1.scatter(npos[0][:,0],npos[0][:,1],color='r')

ax2=pl.subplot2grid((4,4), (0,1))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax2.scatter(npost01[1][:,0],npost01[1][:,1],color='b')

ax3=pl.subplot2grid((4,4), (0,2))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax3.scatter(npost01[0][:,0],npost01[0][:,1],color='b')

ax4=pl.subplot2grid((4,4), (0,3))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax4.scatter(npos[1][:,0],npos[1][:,1],color='r')

ax5=ax4=pl.subplot2grid((4,4), (1,0))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax5.scatter(npost02[1][:,0],npost02[1][:,1],color='b')

ax6=ax4=pl.subplot2grid((4,4), (1,3))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax6.scatter(npost13[1][:,0],npost13[1][:,1],color='b')

ax7=ax4=pl.subplot2grid((4,4), (2,0))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax7.scatter(npost02[0][:,0],npost02[0][:,1],color='b')

ax8=ax4=pl.subplot2grid((4,4), (2,3))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax8.scatter(npost13[0][:,0],npost13[0][:,1],color='b')

ax9=ax4=pl.subplot2grid((4,4), (3,0))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax9.scatter(npos[2][:,0],npos[2][:,1],color='r')

ax10=ax4=pl.subplot2grid((4,4), (3,1))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax10.scatter(npost23[1][:,0],npost23[1][:,1],color='b')

ax11=ax4=pl.subplot2grid((4,4), (3,2))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax11.scatter(npost23[0][:,0],npost23[0][:,1],color='b')

ax12=ax4=pl.subplot2grid((4,4), (3,3))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax12.scatter(npos[3][:,0],npos[3][:,1],color='r')

'''
TEST : SHAPES
'''

import scipy.ndimage as spi

carre=spi.imread('C:/Users/Erwan/Desktop/Stage_2A/Shapes/carre.png').astype(np.float64)/256
rond=spi.imread('C:/Users/Erwan/Desktop/Stage_2A/Shapes/rond.png').astype(np.float64)/256
triangle=spi.imread('C:/Users/Erwan/Desktop/Stage_2A/Shapes/triangle.png').astype(np.float64)/256
fleche=spi.imread('C:/Users/Erwan/Desktop/Stage_2A/Shapes/coeur.png').astype(np.float64)/256



def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0]*I.shape[1],I.shape[2]))

carre=im2mat(carre)
carre=carre[:,2]
carre=carre.reshape((8,8))
pl.imshow(carre)

rond=im2mat(rond)
rond=rond[:,2]
rond=rond.reshape((8,8))
pl.imshow(rond)

triangle=im2mat(triangle)
triangle=triangle[:,2]
triangle=triangle.reshape((8,8))
pl.imshow(triangle)

fleche=im2mat(fleche)
fleche=fleche[:,2]
fleche=fleche.reshape((8,8))
pl.imshow(fleche)


shapes=[carre,rond,triangle,fleche]

S=4
xs=[[] for i in range(S)]


for nb in range(4):
    for i in range(8):
        for j in range(8):
            if shapes[nb][i,j]!=0.99609375:
                xs[nb].append([j,8-i])

xs=np.array([np.array(xs[0]),np.array(xs[1]),np.array(xs[2]),np.array(xs[3])])


fig=pl.figure(figsize=(15,10))

ax1=fig.add_subplot(221)
pl.xlim((0,10))
pl.ylim((0,10))
ax1.scatter(xs[0][:,0],xs[0][:,1],color='b')

ax2=fig.add_subplot(222)
pl.xlim((0,10))
pl.ylim((0,10))
ax2.scatter(xs[1][:,0],xs[1][:,1],color='b')

ax3=fig.add_subplot(223)
pl.xlim((0,10))
pl.ylim((0,10))
ax3.scatter(xs[2][:,0],xs[2][:,1],color='b')

ax4=fig.add_subplot(224)
pl.xlim((0,10))
pl.ylim((0,10))
ax4.scatter(xs[3][:,0],xs[3][:,1],color='b')


ns=[len(xs[s]) for s in range(S)]
N=30

Cs=[sp.spatial.distance.cdist(xs[s],xs[s]) for s in range(S)]
Cs=[cs/cs.max() for cs in Cs]

ps=[ot.unif(ns[s]) for s in range(S)]
p=ot.unif(N)
p=p.reshape(1,len(p)).T
ps=[ps[s].reshape(1,len(ps[s])).T for s in range(S)]


npos=[0,0,0,0]
npos=[mapping.smacof_mds(Cs[s],2) for s in range(S)]

clf = PCA(n_components=2)

npos=[clf.fit_transform(npos[s]) for s in range(S)]

fig=pl.figure(figsize=(15,10))

ax1=fig.add_subplot(221)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax1.scatter(npos[0][:,0],npos[0][:,1],color='b')

ax2=fig.add_subplot(222)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax2.scatter(npos[1][:,0],npos[1][:,1],color='b')

ax3=fig.add_subplot(223)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax3.scatter(npos[2][:,0],npos[2][:,1],color='b')

ax4=fig.add_subplot(224)
pl.xlim((-1,1))
pl.ylim((-1,1))
ax4.scatter(npos[3][:,0],npos[3][:,1],color='b')



lambdast=[[i/3,(3-i)/3] for i in [1,2]]

Ct01=[0 for i in range(2)]
for i in range(2):
    Ct01[i]=gromov.gromov_barycenters(N,[Cs[0],Cs[1]],[ps[0],ps[1]],p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=100,stopThr=1e-3)

Ct02=[0 for i in range(2)]
for i in range(2):
    Ct02[i]=gromov.gromov_barycenters(N,[Cs[0],Cs[2]],[ps[0],ps[2]],p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=100,stopThr=1e-3)

Ct13=[0 for i in range(2)]
for i in range(2):
    Ct13[i]=gromov.gromov_barycenters(N,[Cs[1],Cs[3]],[ps[1],ps[3]],p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=100,stopThr=1e-3)

Ct23=[0 for i in range(2)]
for i in range(2):
    Ct23[i]=gromov.gromov_barycenters(N,[Cs[2],Cs[3]],[ps[2],ps[3]],p,lambdast[i],loss.tensor_square_loss,updates.update_square_loss,5e-4,numItermax=100,stopThr=1e-3)


npost01=[0,0]
npost01=[mapping.smacof_mds(Ct01[s],2) for s in range(2)]
npost01=[clf.fit_transform(npost01[s]) for s in range(2)]

npost02=[0,0]
npost02=[mapping.smacof_mds(Ct02[s],2) for s in range(2)]
npost02=[clf.fit_transform(npost02[s]) for s in range(2)]

npost13=[0,0]
npost13=[mapping.smacof_mds(Ct13[s],2) for s in range(2)]
npost13=[clf.fit_transform(npost13[s]) for s in range(2)]

npost23=[0,0]
npost23=[mapping.smacof_mds(Ct23[s],2) for s in range(2)]
npost23=[clf.fit_transform(npost23[s]) for s in range(2)]



fig=pl.figure(figsize=(10,10))

ax1=pl.subplot2grid((4,4), (0,0))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax1.scatter(npos[0][:,0],npos[0][:,1],color='r')

ax2=pl.subplot2grid((4,4), (0,1))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax2.scatter(npost01[1][:,0],npost01[1][:,1],color='b')

ax3=pl.subplot2grid((4,4), (0,2))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax3.scatter(npost01[0][:,0],npost01[0][:,1],color='b')

ax4=pl.subplot2grid((4,4), (0,3))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax4.scatter(npos[1][:,0],npos[1][:,1],color='r')

ax5=pl.subplot2grid((4,4), (1,0))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax5.scatter(npost02[1][:,0],npost02[1][:,1],color='b')

ax6=pl.subplot2grid((4,4), (1,3))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax6.scatter(npost13[1][:,0],npost13[1][:,1],color='b')

ax7=pl.subplot2grid((4,4), (2,0))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax7.scatter(npost02[0][:,0],npost02[0][:,1],color='b')

ax8=pl.subplot2grid((4,4), (2,3))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax8.scatter(npost13[0][:,0],npost13[0][:,1],color='b')

ax9=pl.subplot2grid((4,4), (3,0))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax9.scatter(npos[2][:,0],npos[2][:,1],color='r')

ax10=pl.subplot2grid((4,4), (3,1))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax10.scatter(npost23[1][:,0],npost23[1][:,1],color='b')

ax11=pl.subplot2grid((4,4), (3,2))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax11.scatter(npost23[0][:,0],npost23[0][:,1],color='b')

ax12=pl.subplot2grid((4,4), (3,3))
pl.xlim((-1,1))
pl.ylim((-1,1))
ax12.scatter(npos[3][:,0],npos[3][:,1],color='r')


'''
TEST : 3D to 2D projection, forced coupling
'''

n=15
mu=np.array([4,4,4])
cov=np.array([[1,0,0],[0,1,0],[0,0,1]])

P=sp.linalg.sqrtm(cov)
xs= np.random.randn(n,3).dot(P)+mu

xt=[[xs[i][0],xs[i][1]] for i in range(len(xs))]
xt=np.asarray(xt)

fig=pl.figure()

ax1=fig.add_subplot(121,projection='3d')

ax1.scatter(xs[:,0],xs[:,1],xs[:,2],color='r')

ax2=fig.add_subplot(122)

ax2.plot(xt[:,0],xt[:,1],'+b',label='Source samples')
pl.show()


p=ot.unif(n)
q=ot.unif(n)
p=p.reshape(1,len(p)).T
q=q.reshape(1,len(q)).T

C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xt,xt)

C1/=C1.max()
C2/=C2.max()

pl.subplot(121)

pl.imshow(C1)

pl.subplot(122)

pl.imshow(C2)


forc=[[i,i] for i in range(10)]


gw=[i for i in range(9)]
for i in range(9):
    gw[i]=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4,forcing=forc[:i])

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4,forcing=forc)
gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4)
pl.imshow(gw,cmap='jet')
pl.colorbar()
pl.show()

fig=pl.figure(figsize=(10,10))

ax1=pl.subplot2grid((3,3), (0,0))
pl.imshow(gw[0],cmap='jet')
pl.title("Nb forcing : 0")

ax2=pl.subplot2grid((3,3), (0,1))
pl.imshow(gw[1],cmap='jet')
pl.title("Nb forcing : 1")

ax3=pl.subplot2grid((3,3), (0,2))
pl.imshow(gw[2],cmap='jet')
pl.title("Nb forcing : 2")

ax4=pl.subplot2grid((3,3), (1,0))
pl.imshow(gw[3],cmap='jet')
pl.title("Nb forcing : 3")

ax5=pl.subplot2grid((3,3), (1,1))
pl.imshow(gw[4],cmap='jet')
pl.title("Nb forcing : 4")

ax6=pl.subplot2grid((3,3), (1,2))
pl.imshow(gw[5],cmap='jet')
pl.title("Nb forcing : 5")

ax7=pl.subplot2grid((3,3), (2,0))
pl.imshow(gw[6],cmap='jet')
pl.title("Nb forcing : 6")

ax8=pl.subplot2grid((3,3), (2,1))
pl.imshow(gw[7],cmap='jet')
pl.title("Nb forcing : 7")

ax9=pl.subplot2grid((3,3), (2,2))
pl.imshow(gw[8],cmap='jet')
pl.title("Nb forcing : 8")


import imp
imp.reload(loss)
imp.reload(gromov)
imp.reload(updates)
imp.reload(mapping)



