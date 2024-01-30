#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alblle
a.llera@donders.ru.nl
"""

#Import dependences and create functions

import numpy as np
#import matplotlib.pyplot as plt
import copy as cp
import time
from scipy.linalg import eigh




def basic_csp(C1=None,C2=None):
    #inputs: C1 and C2, 2 covariance matrices of same size
    #outputs: csp_eigvecs, projected_eigvals1, projected_eigvals2, (using diag(w'*C*w))

    eigvals, csp_eigvecs = eigh(C1, C1+C2, eigvals_only=False)

    projected_eigvals1 = np.diag(np.dot(np.dot(np.transpose(csp_eigvecs),C1),csp_eigvecs))
    projected_eigvals2 = np.diag(np.dot(np.dot(np.transpose(csp_eigvecs),C2),csp_eigvecs))

    return csp_eigvecs, projected_eigvals1, projected_eigvals2


def features_csp(csp_eigvecs=None,covariances=None,nolog=False):
#inputs: csp_eigvecs is output of basic_csp, 
#        covariances is a 3d numpy array of covariance matrices,with first dimension de number of covariances
#output: log-variance of data projections to csp_eigvecs

    features=np.ndarray(shape=(covariances.shape[0],covariances.shape[1]))
    for i in range(covariances.shape[0]):
        if nolog==False:
            features[i,:]= np.log(np.diag(np.dot(np.dot(np.transpose(csp_eigvecs),np.squeeze(covariances[i,:,:])),csp_eigvecs)))
        else:
            features[i,:]= np.diag(np.dot(np.dot(np.transpose(csp_eigvecs),np.squeeze(covariances[i,:,:])),csp_eigvecs))

    return features

def cv_classification_spade(all_covs=None,all_labels=None,Nfolds=10):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    spade_acc=np.zeros([Nfolds])


    
    #create folds
    fold_indices = np.arange(all_covs.shape[0])
    np.random.shuffle(fold_indices)
    eval_indices = np.array_split(fold_indices, Nfolds)
    fold=-1
    
    for e in eval_indices:
        fold+=1
        #Define the evaluation set for the current fold
        eval_set = all_covs[e,:,:]
        eval_labels=  all_labels[e]

        #exclude the upon parts indices from the 
        #whole array (similarly on the upon answers)
        mask_eval = np.ones(all_covs.shape[0], bool)
        #Set indices of the eval set to false
        mask_eval[e] = False
        #Subset by the bool array:
        train_set = all_covs[mask_eval,:,:]
        train_labels=all_labels[mask_eval]

    
    
    #--------spade
        C1=np.mean(train_set[train_labels==0,:,:],axis=0)
        C2=np.mean(train_set[train_labels==1,:,:],axis=0)
        csp_eigvecs, projected_eigvals1, projected_eigvals2 = basic_csp(C1,C2)
        csp_train_features=features_csp(csp_eigvecs,train_set)
        csp_train_features=np.matrix(csp_train_features[:,-1]).T#[:,[0,-1]]
        csp_eval_features=features_csp(csp_eigvecs,eval_set)
        csp_eval_features=np.matrix(csp_eval_features[:,-1]).T #[0,-1]]
        clf = LinearDiscriminantAnalysis()
        clf.fit(np.asarray(csp_train_features), train_labels)

        pred=clf.predict(np.asarray(csp_eval_features))
        spade_acc[fold]=np.mean(pred==eval_labels)
        
    return spade_acc
    



