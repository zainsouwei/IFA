#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alblle
a.llera@donders.ru.nl
"""

import random
import numpy as np
from sklearn.covariance import LedoitWolf

def distance_riemann(A, B):
    r"""Affine-invariant Riemannian distance between two SPD matrices.

    Compute the affine-invariant Riemannian distance between two SPD matrices A
    and B:

    .. math::
        d(A,B) = {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}

    where :math:`\lambda_i` are the joint eigenvalues of A and B.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        First SPD matrix.
    B : ndarray, shape (n, n)
        Second SPD matrix.

    Returns
    -------
    d : float
        Affine-invariant Riemannian distance between A and B.
    """
    return np.sqrt((np.log(eigvalsh(A, B))**2).sum())



def generated_SPD_4x4(dim=4,l=0.9,b=0.5,plotme=0):
    #l=0 between 0 and 1 is the similarity between the 2 covariances generated
    #I am generating processes from a multivartiate gaussian of dimensions 4x4 as reference signals
    p=4
    n=500
    #generate 2 basic signals
    signal1=np.random.normal(0,1,n)
    signal2=np.random.normal(0,1,n)


    #first easy try: half channels one signal, half another signal
    s1=np.zeros([p,n])
    for i in range(p):
        s1[0,:]=signal1
        #b=0.1 #random.uniform(0, 1)
        s1[1,:]=np.multiply(signal1,b)  + np.multiply(np.random.normal(0,1,n),1-b) 

        s1[2,:]=signal2
        #b= 0.1# random.uniform(0, 1)
        s1[3,:]=-(np.multiply(signal2,b)  + np.multiply(np.random.normal(0,1,n),1-b) )


    s2=s1[[0,2,1,3],:]
    
    
    if dim>4:
        s1=np.concatenate([s1,np.random.multivariate_normal(np.zeros(dim-p),np.eye(dim-p),n).T],axis=0)
        s2=np.concatenate([s2,np.random.multivariate_normal(np.zeros(dim-p),np.eye(dim-p),n).T],axis=0)


    #from sklearn.preprocessing import scale
    #s1=scale(s1,axis=1,with_mean=True, with_std=True, copy=True)
    #print(np.mean(s1,axis=1))
    #s2=scale(s2,axis=1,with_mean=True, with_std=True, copy=True)
    #print(np.mean(s2,axis=1))

    #Deviant covariance  cov1          
    cov1= np.cov(s1)   

    #Normative covariance cov2
    cov2= np.cov(s2) 
    
    cov2= ((1-l) * cov2) + (l*cov1)

    
    
    if plotme==1:
        fig,axes=plt.subplots(1,2)        
        im=axes[0].imshow(cov1,vmin=-1, vmax=1)
        fig.colorbar(im,ax=axes[0])

        im2=axes[1].imshow(cov2,vmin=-1, vmax=1)
        fig.colorbar(im2,ax=axes[1])

    if 0:
        scipy.io.savemat('cov1.mat', mdict={'cov1': cov1})
        scipy.io.savemat('cov2.mat', mdict={'cov2': cov2})



    #csp_eigvecs, projected_eigvals1, projected_eigvals2 = basic_csp(cov1,cov2)
    ##print(projected_eigvals1  + projected_eigvals2) # this adds to one for every direction
    #print('Diference (abs) variances across groups', np.abs(projected_eigvals1-projected_eigvals2))

    #print('Riemannian distance between Normative and deviat matrices is',distance_riemann(cov1, cov2))
    #print(cov1)
    #print(cov2)
    
    return cov1,cov2#,distance_riemann(cov1, cov2),np.abs(projected_eigvals1-projected_eigvals2)

def generate_time_series(Nsubjects,timepoints,cov):
    #generate subjects data
    p=np.shape(cov)[0]
    mean = np.zeros(p)
    data=np.ndarray(shape=(Nsubjects,p,timepoints))
    covs=np.ndarray(shape=(Nsubjects,p,p))
    for sub in range(Nsubjects):
        #generate each subject data for both conditions
        tmp1=np.random.multivariate_normal(mean,cov,timepoints).T
        #demean, maybe not needed
        tmp1=tmp1-np.matrix(np.mean(tmp1,1)).T
        data[sub,:,:]=tmp1
        #Get each subject spatial covariance
        #if need to regularize use Ledoit Wolf sklearn.covariance.LedoitWolf instead of np.cov
        Lcov = LedoitWolf().fit(np.squeeze(data[sub,:,:]).T)
        covs[sub,:,:]=Lcov.covariance_
        #covs[sub,:,:]=np.cov(np.squeeze(data[sub,:,:]))
        
    return data,covs
  
def generate_spectrum(Normative_cov,deviant_cov,Nsubjects=100,p=4,timepoints=500):
    data1=np.ndarray(shape=(Nsubjects,p,timepoints))
    covs1=np.ndarray(shape=(Nsubjects,p,p))
    lambd=np.linspace(0,1,Nsubjects)
    for sub in range(Nsubjects):
        #generate each subject data for both conditions
        spectrum_cov1= ((1-lambd[sub]) * deviant_cov) + (lambd[sub]*Normative_cov)
        deviants_data,deviants_covs=generate_time_series(Nsubjects=2,timepoints=500,cov=spectrum_cov1)
        data1[sub,:,:]=deviants_data[0,:,:]
        covs1[sub,:,:]=deviants_covs[0,:,:]
        
    return data1,covs1





