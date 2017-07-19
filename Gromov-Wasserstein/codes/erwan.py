


import sys
sys.path.append('C:/Users/Erwan/Desktop/Stage_2A/Gromov-Wasserstein/codes/gw')
sys.path.append('C:/Users/Erwan/Desktop/Stage_2A/Gromov-Wasserstein/codes')



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
import sklearn
import scipy.optimize as spo
import kcca


import imp
imp.reload(loss)
imp.reload(gromov)
imp.reload(updates)
imp.reload(mapping)


#passe d'une liste de label [1 2 1 3 ...] à une matrice [[1 0 0] [0 1 0] [1 0 0] [0 0 1] ...]

def vectorizeLabels(y):
    vals=np.unique(y)
    Y=np.zeros((len(y),len(vals)))
    for i,val in enumerate(vals):
        Y[:,i]=(y==val)
    return Y

# SVM classifier
    

def hinge_squared_reg(w,X,Y,lambd):
    """
    compute loss dans gradient for squared hing loss with quadratic regularization

    """
    nbclass=Y.shape[1]
    w=w.reshape((X.shape[0],Y.shape[1]))
    f=X.dot(w)

    err_alpha=np.maximum(0,1-f)
    err_alpha1=np.maximum(0,1+f)

    loss=0
    grad=np.zeros_like(w)
    for i in range(nbclass):
        loss+=Y[:,i].T.dot(err_alpha[:,i]**2)+(1-Y[:,i]).T.dot(err_alpha1[:,i]**2)
        grad[:,i]+=2*X.T.dot(-Y[:,i]*err_alpha[:,i]+(1-Y[:,i])*err_alpha1[:,i]) # alpha

    # regularization term
    loss+=lambd*np.sum(w**2)/2
    grad+=lambd*w

    return loss,grad.ravel()
  
def hinge_squared_reg_bias(w,X,Y,lambd):
    """
    compute loss dans gradient for squared hing loss with quadratic regularization

    """
    nbclass=Y.shape[1]
    w=w.reshape((X.shape[1],Y.shape[1]))
    f=X.dot(w)

    err_alpha=np.maximum(0,1-f)
    err_alpha1=np.maximum(0,1+f)

    loss=0
    grad=np.zeros_like(w)
    for i in range(nbclass):
        loss+=Y[:,i].T.dot(err_alpha[:,i]**2)+(1-Y[:,i]).T.dot(err_alpha1[:,i]**2)
        grad[:,i]+=2*X.T.dot(-Y[:,i]*err_alpha[:,i]+(1-Y[:,i])*err_alpha1[:,i]) # alpha

    # regularization term
    w[:,-1]=0
    loss+=lambd*np.sum(w**2)/2
    grad+=lambd*w

    return loss,grad.ravel()


class SVMClassifier(object):

    def __init__(self,lambd=1e-2,bias=False):
        self.lambd=lambd
        self.w=None
        self.bias=bias


    def fit(self,K,y):  
        if self.bias:
            K1=np.hstack((K,np.ones((K.shape[0],1))))
            self.w=np.zeros((K1.shape[1],y.shape[1]))
            self.w,self.f,self.log=spo.fmin_l_bfgs_b(lambda w: hinge_squared_reg_bias(w,X=K1,Y=y,lambd=self.lambd),self.w,maxiter=1000,maxfun=1000)            
            self.b=self.w.reshape((K1.shape[1],y.shape[1]))[-1,:]
            self.w=self.w.reshape((K1.shape[1],y.shape[1]))[:-1,:]

        else:
            self.w=np.zeros((K.shape[1],y.shape[1]))
            self.w,self.f,self.log=spo.fmin_l_bfgs_b(lambda w: hinge_squared_reg(w,X=K,Y=y,lambd=self.lambd),self.w,maxiter=1000,maxfun=1000)            
            self.w=self.w.reshape((K.shape[1],y.shape[1]))

    def predict(self,K):
        if self.bias:
            return np.dot(K,self.w)+self.b
        else:
            return np.dot(K,self.w)
    

            
            
#################################################################
# EXEMPLE d'utilisation
       #--------------------II. prepare data--------------------------------------
        # ENTRAINEMENT
        Xs = data[src][0]  
        Ys = data[src][1]
        Ys=vectorizeLabels(Ys)
        # TEST
        Xt = data[tgt][0]
        Yt = data[tgt][1]

        #--------------------III. run experiments----------------------------------
         #------ SVM standard
        # compute kernels 
        

        # gamma/lambda paramètres du classifieur
        
        clf = SVMClassifier(lambd)
        
        
        K=sklearn.metrics.pairwise.rbf_kernel(Xs,gamma=gamma)
        Kt=sklearn.metrics.pairwise.rbf_kernel(Xs,Xt,gamma=gamma)

        clf.fit(K,Ys)
        ypred=clf.predict(Kt.T)        
        ydec=np.argmax(ypred,1)+1        
        print 'base=',np.mean(ydec==Yt)*100
        res.append(np.mean(ydec==Yt)*100)

        
##################################################################
# Propagation de labels

# calcul du transport --> dans ton cas ca sera GW
G=  ot.emd(wa,wb,C)

# ntest nombre de points en target
# y labels en source sous forme VECTORISE

Yst=ntest*G.T.dot(y)

'''
TEST
'''

np.random.seed(0) # makes example reproducible

n=100 # nb samples in source and target datasets
theta=2*np.pi/20
nz=0.1
xs,ysour=ot.datasets.get_data_classif('gaussrot',n,nz=nz)
xt,yt=ot.datasets.get_data_classif('gaussrot',n,theta=theta,nz=nz)
xt[yt==2]*=3
xt=xt+4


pl.figure(1,(8,5))
pl.clf()
pl.scatter(xs[:,0],xs[:,1],c=ysour,marker='+',label='Source samples')
pl.scatter(xt[:,0],xt[:,1],c=ydec,marker='o',label='Target samples')

ys=vectorizeLabels(ysour)


clf = SVMClassifier(0.1)

K=sklearn.metrics.pairwise.rbf_kernel(xs,gamma=1)
Kt=sklearn.metrics.pairwise.rbf_kernel(xs,xt,gamma=1)

clf.fit(K,ys)
ypred=clf.predict(Kt.T)
ydec=np.argmax(ypred,1)+1


'''
TEST : 2d 2d spread
'''

p=ot.unif(n)
q=ot.unif(n)
p=p.reshape(1,len(p)).T
q=q.reshape(1,len(q)).T

C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xt,xt)

C1/=C1.max()
C2/=C2.max()

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4)

ysp=n*gw.T.dot(ys)
yspdec=np.argmax(ysp,1)+1

pl.figure(1,(8,5))
pl.clf()
pl.scatter(xs[:,0],xs[:,1],c=ysour,marker='+',label='Source samples')
pl.scatter(xt[:,0],xt[:,1],c=yspdec,marker='o',label='Target samples')

'''
TEST : 2d 3d spread 2 classes
'''

#data generation

np.random.seed(0) # makes example reproducible

n=100 # nb samples in source and target datasets
theta=2*np.pi/3
nz=0.1
xs,ys=ot.datasets.get_data_classif('gaussrot',n,nz=nz)
xt,yt=ot.datasets.get_data_classif('gaussrot',n,theta=theta,nz=nz)
z=np.random.randn(n,1)-0.5
xt=np.hstack((xt,z))


#sample plotting

fig=pl.figure()

ax1=fig.add_subplot(121)

ax1.scatter(xs[:,0],xs[:,1],c=ys,marker='+')

ax2=fig.add_subplot(122,projection='3d')

ax2.scatter(xt[:,0],xt[:,1],xt[:,2],color='r')

#weigths and metrics

p=ot.unif(n)
q=ot.unif(n)
p=p.reshape(1,len(p)).T
q=q.reshape(1,len(q)).T

C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xt,xt)

C1/=C1.max()
C2/=C2.max()

# metrics plotting

pl.subplot(121)

pl.imshow(C1)

pl.subplot(122)

pl.imshow(C2)

#gromov wasserstein

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4)

#transport plotting

pl.imshow(gw,cmap='jet')
pl.colorbar()
pl.show()

#label spreading

ysVec=vectorizeLabels(ys)
ysp=n*gw.T.dot(ysVec)
yspdec=np.argmax(ysp,1)+1

print("base=",np.mean(yspdec==yt)*100)
res.append(np.mean(ydec==yt)*100)


#plotting labels

fig=pl.figure()

ax1=fig.add_subplot(121)

ax1.scatter(xs[:,0],xs[:,1],c=ys,marker='+')
pl.title("Source samples")

ax2=fig.add_subplot(122,projection='3d')

ax2.scatter(xt[:,0],xt[:,1],xt[:,2],c=yspdec)
pl.title("Target samples")


'''
TEST : 2d 3d 3 classes
'''

#data generation

np.random.seed(0) # makes example reproducible

n=90 # nb samples in source and target datasets

mus1=np.array([2.5,2.5])
covs1=np.array([[0.2,0],[0,0.2]])
Ps1=sp.linalg.sqrtm(covs1)
xs1=np.random.randn(int(n/3),2).dot(Ps1)+mus1
ys1=np.array([1 for i in range(int(n/3))])

mus2=np.array([-1.5,3])
covs2=np.array([[0.6,0],[0,0.6]])
Ps2=sp.linalg.sqrtm(covs2)
xs2=np.random.randn(int(n/3),2).dot(Ps2)+mus2
ys2=np.array([2 for i in range(int(n/3))])

mus3=np.array([1,1])
covs3=np.eye(2)
Ps3=sp.linalg.sqrtm(covs3)
xs3=np.random.randn(int(n/3),2).dot(Ps3)*mus3
ys3=np.array([3 for i in range(int(n/3))])

xs=np.concatenate((xs1,xs2,xs3))
ys=np.concatenate((ys1,ys2,ys3))
############################################
mut1=np.array([2.5,2.5,2.5])
covt1=np.array([[0.2,0,0],[0,0.2,0],[0,0,0.2]])
Pt1=sp.linalg.sqrtm(covt1)
xt1=np.random.randn(int(n/3),3).dot(Pt1)+mut1
yt1=np.array([1 for i in range(int(n/3))])

mut2=np.array([-1.5,3,0.5])
covt2=np.array([[0.6,0,0],[0,0.6,0],[0,0,0.6]])
Pt2=sp.linalg.sqrtm(covt2)
xt2=np.random.randn(int(n/3),3).dot(Pt2)+mut2
yt2=np.array([2 for i in range(int(n/3))])

mut3=np.array([1,1,1])
covt3=np.eye(3)
Pt3=sp.linalg.sqrtm(covt3)
xt3=np.random.randn(int(n/3),3).dot(Pt3)*mut3
yt3=np.array([3 for i in range(int(n/3))])

xt=np.concatenate((xt1,xt2,xt3))
yt=np.concatenate((yt1,yt2,yt3))



#sample plotting

fig=pl.figure()

ax1=fig.add_subplot(121)

ax1.scatter(xs[:,0],xs[:,1],c=ys,marker='+')

ax2=fig.add_subplot(122,projection='3d')

ax2.scatter(xt[:,0],xt[:,1],xt[:,2],c=yspdec)

#weigths and metrics

p=ot.unif(n)
q=ot.unif(n)
p=p.reshape(1,len(p)).T
q=q.reshape(1,len(q)).T

C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xt,xt)

C1/=C1.max()
C2/=C2.max()

# metrics plotting

pl.subplot(121)

pl.imshow(C1)

pl.subplot(122)

pl.imshow(C2)

#gromov wasserstein

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4)

#transport plotting

pl.imshow(gw,cmap='jet')
pl.colorbar()
pl.show()

#label spreading

ysVec=vectorizeLabels(ys)
ysp=n*gw.T.dot(ysVec)
yspdec=np.argmax(ysp,1)+1

print("base=",np.mean(yspdec==yt)*100)


#plotting labels

fig=pl.figure()

ax1=fig.add_subplot(121)

ax1.scatter(xs[:,0],xs[:,1],c=ys,marker='+')
pl.title("Source samples")

ax2=fig.add_subplot(122,projection='3d')

ax2.scatter(xt[:,0],xt[:,1],xt[:,2],c=yspdec)
pl.title("Target samples")


'''
TEST : 8d 2d 3 classes
'''

#data generation

np.random.seed(0) # makes example reproducible

n=90 # nb samples in source and target datasets

mus1=np.array([2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5])
covs1=np.eye(8)*0.2
Ps1=sp.linalg.sqrtm(covs1)
xs1=np.random.randn(int(n/3),8).dot(Ps1)+mus1
ys1=np.array([1 for i in range(int(n/3))])

mus2=np.array([-1.5,3,1,1,1,1,1,1])
covs2=np.eye(8)*0.6
Ps2=sp.linalg.sqrtm(covs2)
xs2=np.random.randn(int(n/3),8).dot(Ps2)+mus2
ys2=np.array([2 for i in range(int(n/3))])

mus3=np.array([1,1,1,1,1,1,1,1])
covs3=np.eye(8)
Ps3=sp.linalg.sqrtm(covs3)
xs3=np.random.randn(int(n/3),8).dot(Ps3)*mus3
ys3=np.array([3 for i in range(int(n/3))])

xs=np.concatenate((xs1,xs2,xs3))
ys=np.concatenate((ys1,ys2,ys3))
############################################
mut1=np.array([2.5,2.5])
covt1=np.eye(2)*0.2
Pt1=sp.linalg.sqrtm(covt1)
xt1=np.random.randn(int(n/3),2).dot(Pt1)+mut1
yt1=np.array([1 for i in range(int(n/3))])

mut2=np.array([-1.5,3])
covt2=np.eye(2)*0.6
Pt2=sp.linalg.sqrtm(covt2)
xt2=np.random.randn(int(n/3),2).dot(Pt2)+mut2
yt2=np.array([2 for i in range(int(n/3))])

mut3=np.array([1,1])
covt3=np.eye(2)
Pt3=sp.linalg.sqrtm(covt3)
xt3=np.random.randn(int(n/3),2).dot(Pt3)*mut3
yt3=np.array([3 for i in range(int(n/3))])

xt=np.concatenate((xt1,xt2,xt3))
yt=np.concatenate((yt1,yt2,yt3))

#weigths and metrics

p=ot.unif(n)
q=ot.unif(n)
p=p.reshape(1,len(p)).T
q=q.reshape(1,len(q)).T

C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xt,xt)

C1/=C1.max()
C2/=C2.max()

# metrics plotting

pl.subplot(121)

pl.imshow(C1)

pl.subplot(122)

pl.imshow(C2)

#gromov wasserstein

gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4)

#transport plotting

pl.imshow(gw,cmap='jet')
pl.colorbar()
pl.show()

#label spreading

ysVec=vectorizeLabels(ys)
ysp=n*gw.T.dot(ysVec)
yspdec=np.argmax(ysp,1)+1

print("base=",np.mean(yspdec==yt)*100)


'''
TEST : USPS vs MNIST
'''

from scipy.io import loadmat
usps=loadmat("C:/Users/Erwan/Desktop/Stage_2A/Documents/USPS_vs_MNIST")
mnist=loadmat("C:/Users/Erwan/Desktop/Stage_2A/Documents/2k2k")



Xusps=usps['X_src'].T
Yusps=usps['Y_src']
Xmnist=mnist['fea']
Ymnist=mnist['gnd']+1


def select_usps(y):
    xs=[i for i in range(len(y))]
    ys=[i for i in range(len(y))]
    for i in range(len(y)):
        indices=(Yusps==y[i]).nonzero()
        xs[i]=Xusps[indices[0]][:126]
        ys[i]=Yusps[indices[0]][:126]
    return np.concatenate(xs),np.concatenate(np.concatenate(ys))

def select_usps2(y):
    xs=[i for i in range(len(y))]
    ys=[i for i in range(len(y))]
    for i in range(len(y)):
        indices=(Yusps==y[i]).nonzero()
        xs[i]=Xusps[indices[0]][:63]
        ys[i]=Yusps[indices[0]][:63]
    return np.concatenate(xs),np.concatenate(np.concatenate(ys))

def select_mnist(y):
    xttr=[i for i in range(len(y))]
    yttr=[i for i in range(len(y))]
    xtte=[i for i in range(len(y))]
    ytte=[i for i in range(len(y))]
    for i in range(len(y)):
        indices=(Ymnist==y[i]).nonzero()
        xttr[i]=Xmnist[indices[0]][:63]
        yttr[i]=Ymnist[indices[0]][:63]
        xtte[i]=Xmnist[indices[0]][63:126]
        ytte[i]=Ymnist[indices[0]][63:126]
    return np.concatenate(xttr),np.concatenate(np.concatenate(yttr)),np.concatenate(xtte),np.concatenate(np.concatenate(ytte))

def affiche(x,i):
    if x.shape[1]==256:
        pl.imshow(x[i].reshape(16,16))
    if x.shape[1]==784:
        pl.imshow(x[i].reshape(28,28))
        
        
y=[1,2,3,4,5,6,7,8,9,10]


xs,ys=select_usps(y)
xs2,ys2=select_usps2(y)
xttr,yttr,xtte,ytte=select_mnist(y)

p=ot.unif(xs.shape[0])
q=ot.unif(xttr.shape[0])
p=p.reshape(1,len(p)).T
q=q.reshape(1,len(q)).T

C1=sp.spatial.distance.cdist(xs,xs)

C2=sp.spatial.distance.cdist(xttr,xttr)

C1/=C1.max()
C2/=C2.max()

#GEODESIC DISTANCE
geo=sklearn.manifold.Isomap(path_method='auto')
geo.fit(xs)
C1=geo.dist_matrix_
C1/=C1.max()
geo.fit(xttr)
C2=geo.dist_matrix_
C2/=C2.max()

#TEST FORCING


def forcing(nb_k):
    class_forcing=np.zeros((len(xs),len(xttr)))
    for i in range(len(y)):
        k=np.random.randint(63*i,63*(i+1),size=(nb_k,1))
        for j in range(class_forcing.shape[0]):
            if j<126*i or j>=126*(i+1):
                for pk in k:
                    class_forcing[j][pk]=np.inf
    return class_forcing

class_forcing=forcing(1) 
       
pl.imshow(class_forcing,cmap='jet')

'''
pl.subplot(121)
pl.imshow(C1)
pl.subplot(122)
pl.imshow(C2)
'''

gw=gromov.class_gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4,class_forcing)
gw=gromov.gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,5e-4)

    
ysVec=vectorizeLabels(ys)
ysp=len(xttr)*gw.T.dot(ysVec)

clf = SVMClassifier()
K=sklearn.metrics.pairwise.rbf_kernel(xttr,gamma=1e-7)
Kt=sklearn.metrics.pairwise.rbf_kernel(xttr,xtte,gamma=1e-7)
clf.fit(K,ysp)
ypred=clf.predict(Kt.T)        
 
ydirmax=np.array([y[i] for i in np.argmax(ysp,1)])
yspdec=np.array([y[i] for i in np.argmax(ypred,1)])
ytknn=-np.ones(len(xttr))
for i in range(10):
    ytknn[i*63]=yttr[i*63]
from sklearn.semi_supervised import label_propagation
label_spread=label_propagation.LabelSpreading(kernel='knn',alpha=1.0)
label_spread.fit(xttr,ytknn)
ypredknn=label_spread.transduction_
print("Direct Argmax =",np.mean(ydirmax==yttr)*100)
print("SVM Classifier =",np.mean(yspdec==ytte)*100)
print("Label Spread knn =",np.mean(ypredknn==yttr)*100)



clf1 = PCA(n_components=2)
npos1=clf1.fit_transform(xs)
npot1=clf1.fit_transform(xttr)

clf2 = PCA(n_components=3)
npos2=clf2.fit_transform(xs)
npot2=clf2.fit_transform(xttr)


fig=pl.figure(figsize=(15,15))
ax1=fig.add_subplot(231)
ax1.scatter(npos1[:,0],npos1[:,1],c=ys,s=10,cmap='jet')
pl.title("Xs : PCA projection (2D)")
ax2=fig.add_subplot(232)
ax2.scatter(npot1[:,0],npot1[:,1],c=yttr,s=10,cmap='jet')
pl.title("Xttr : PCA projection (2D) (True Labels)")
ax3=fig.add_subplot(233)
ax3.scatter(npot1[:,0],npot1[:,1],c=ydirmax,s=30,marker='+',cmap='jet')
pl.title("Xttr : PCA projection (2D) (Interp)")
ax4=fig.add_subplot(234,projection='3d')
ax4.scatter(npos2[:,0],npos2[:,1],npos2[:,2],c=ys,s=10,cmap='jet')
pl.title("Xs : PCA projection (3D)")
ax5=fig.add_subplot(235,projection='3d')
ax5.scatter(npot2[:,0],npot2[:,1],npot2[:,2],c=yttr,s=10,cmap='jet')
pl.title("Xttr : PCA projection (3D) (True Labels)")
ax6=fig.add_subplot(236,projection='3d')
ax6.scatter(npot2[:,0],npot2[:,1],npot2[:,2],c=ydirmax,s=30,marker='+',cmap='jet')
pl.title("Xttr : PCA projection (3D) (Interp)")


confus_mat=sklearn.metrics.confusion_matrix(yttr,ydirmax,y)
pl.imshow(confus_mat)
pl.colorbar() 

xsInt=gw.dot(xttr)

fig=pl.figure(figsize=(15,15))
ax1=pl.subplot2grid((10,2),(0,0))
affiche(xs,0)

inter=5*np.logspace(-4,2,10)
res=[]

resSauv=[61.03,56.44,26.51,24.01,23.98,24.33,23.76,23.06,23.79,23.61]

for i in range(10):
    resi=[]
    for j in range(10):
        class_forcing=forcing(1)
        gw=gromov.class_gromov_wasserstein(C1,C2,p,q,loss.tensor_square_loss,inter[i],class_forcing)
        ysVec=vectorizeLabels(ys)
        ysp=len(xttr)*gw.T.dot(ysVec)
        ydirmax=np.array([y[i] for i in np.argmax(ysp,1)])
        resi.append(np.mean(ydirmax==yttr)*100)
        print(" i : ",i," j : ",j)
    res.append(np.mean(resi))
        
pl.plot(inter,res)

#TEST KCCA#

kcca=kcca.KCCA(n_components=10,kernel='rbf',n_jobs=1,epsilon=0.1).fit(xs,xttr)



ytknn=-np.ones(len(xttr))

for i in range(10):
    ytknn[i*63]=yttr[i*63]

from sklearn.semi_supervised import label_propagation
label_spread=label_propagation.LabelSpreading(kernel='knn',alpha=1.0)
label_spread.fit(xttr,ytknn)
ypredknn=label_spread.transduction_

print("Label Spread knn =",np.mean(ypredknn==yttr)*100)
            

#TEST#

def test(y):
    score_arg_eucl_nf=[]
    score_class_eucl_nf=[]
    score_arg_geod_nf=[]
    score_class_geod_nf=[]
    score_arg_eucl_f=[]
    score_class_eucl_f=[]
    score_arg_geod_f=[]
    score_class_geod_f=[]
    geo=sklearn.manifold.Isomap(path_method='auto')
    for i in range(10):
        xs,ys=select_usps(y)
        xttr,yttr,xtte,ytte=select_mnist(y)
        p=ot.unif(xs.shape[0])
        q=ot.unif(xttr.shape[0])
        p=p.reshape(1,len(p)).T
        q=q.reshape(1,len(q)).T

        C1_eucl=sp.spatial.distance.cdist(xs,xs)

        C2_eucl=sp.spatial.distance.cdist(xttr,xttr)

        C1_eucl/=C1_eucl.max()
        C2_eucl/=C2_eucl.max()
        
        geo.fit(xs)
        C1_geod=geo.dist_matrix_
        C1_geod/=C1_geod.max()
        geo.fit(xttr)
        C2_geod=geo.dist_matrix_
        C2_geod/=C2_geod.max()
        
        class_forcing=forcing(1)
        
        gw_eucl_nf=gromov.gromov_wasserstein(C1_eucl,C2_eucl,p,q,loss.tensor_square_loss,5e-4)
        gw_eucl_f=gw=gromov.class_gromov_wasserstein(C1_eucl,C2_eucl,p,q,loss.tensor_square_loss,5e-4,class_forcing)
        gw_geod_nf=gromov.gromov_wasserstein(C1_geod,C2_geod,p,q,loss.tensor_square_loss,5e-4)
        gw_geod_f=gw=gromov.class_gromov_wasserstein(C1_geod,C2_geod,p,q,loss.tensor_square_loss,5e-4,class_forcing)
        
        ysVec=vectorizeLabels(ys)
        ysp_eucl_nf=len(xttr)*gw_eucl_nf.T.dot(ysVec)
        ysp_eucl_f=len(xttr)*gw_eucl_f.T.dot(ysVec)
        ysp_geod_nf=len(xttr)*gw_geod_f.T.dot(ysVec)
        ysp_geod_f=len(xttr)*gw_geod_f.T.dot(ysVec)

        clf = SVMClassifier()
        K=sklearn.metrics.pairwise.rbf_kernel(xttr,gamma=1e-7)
        Kt=sklearn.metrics.pairwise.rbf_kernel(xttr,xtte,gamma=1e-7)
        
        clf.fit(K,ysp_eucl_nf)
        ypred=clf.predict(Kt.T)        
        ydirmax=np.array([y[i] for i in np.argmax(ysp_eucl_nf,1)])
        yspdec=np.array([y[i] for i in np.argmax(ypred,1)])
        score_arg_eucl_nf.append(np.mean(ydirmax==yttr)*100)
        score_class_eucl_nf.append(np.mean(yspdec==ytte)*100)
        
        clf.fit(K,ysp_eucl_f)
        ypred=clf.predict(Kt.T)        
        ydirmax=np.array([y[i] for i in np.argmax(ysp_eucl_f,1)])
        yspdec=np.array([y[i] for i in np.argmax(ypred,1)])
        score_arg_eucl_f.append(np.mean(ydirmax==yttr)*100)
        score_class_eucl_f.append(np.mean(yspdec==ytte)*100)
        
        clf.fit(K,ysp_geod_nf)
        ypred=clf.predict(Kt.T)        
        ydirmax=np.array([y[i] for i in np.argmax(ysp_geod_nf,1)])
        yspdec=np.array([y[i] for i in np.argmax(ypred,1)])
        score_arg_geod_nf.append(np.mean(ydirmax==yttr)*100)
        score_class_geod_nf.append(np.mean(yspdec==ytte)*100)
        
        clf.fit(K,ysp_geod_f)
        ypred=clf.predict(Kt.T)        
        ydirmax=np.array([y[i] for i in np.argmax(ysp_geod_f,1)])
        yspdec=np.array([y[i] for i in np.argmax(ypred,1)])
        score_arg_geod_f.append(np.mean(ydirmax==yttr)*100)
        score_class_geod_f.append(np.mean(yspdec==ytte)*100)
    print("Euclidean Distance, non_forcing : Argmax :",np.mean(score_arg_eucl_nf)," SVM :",np.mean(score_class_eucl_nf))
    print("Euclidean Distance, forcing : Argmax :",np.mean(score_arg_eucl_f)," SVM :",np.mean(score_class_eucl_f))
    print("Geodesic Distance, non_forcing : Argmax :",np.mean(score_arg_geod_nf)," SVM :",np.mean(score_class_geod_nf))
    print("Geodesic Distance, forcing : Argmax :",np.mean(score_arg_geod_f)," SVM :",np.mean(score_class_geod_f))

test(y)

