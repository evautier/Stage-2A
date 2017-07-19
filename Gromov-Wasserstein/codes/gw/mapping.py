
import numpy as np
from sklearn import manifold



def smacof_mds(C,dim,maxIter=3000,eps=1e-9):
    """
    Returns an interpolated point cloud following the dissimilarity matrix C using SMACOF
    multidimensional scaling (MDS) in specific dimensionned target space
    
    Parameters
    ----------
    C : np.ndarray(ns,ns)
        dissimilarity matrix
    dim : Integer
          dimension of the targeted space
    maxIter : Maximum number of iterations of the SMACOF algorithm for a single run

    eps : relative tolerance w.r.t stress to declare converge


    Returns
    -------
    npos : R**dim ndarray
           Embedded coordinates of the interpolated point cloud (defined with one isometry)
         
    
    """
    
    seed=np.random.RandomState(seed=3)
        
    mds=manifold.MDS(dim,max_iter=3000,eps=1e-9,dissimilarity='precomputed',n_init=1)
    pos=mds.fit(C).embedding_
        
    nmds = manifold.MDS(2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", random_state=seed, n_init=1)
    npos = nmds.fit_transform(C, init=pos)  

    return npos
    


