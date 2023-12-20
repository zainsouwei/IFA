
import os 
import sys
import numpy as np

#modify next line to provide the  path to your spade directory
toolbox_path = r"C:\Users\zaisou\Desktop\spade_zain\spade_tools_python"





sys.path.append(os.path.abspath(toolbox_path)) 
# from spade_tools_python import spade_tools, generate_spade_toys
from spade_tools import basic_csp,cv_classification_spade
from generate_spade_toys import generated_SPD_4x4, generate_time_series




#EXAMPLE 1: run spade between 2 covs

#generate 2 random covariance matrices, cov 1 and cov 2
l=0.9
b=0.3
cov1,cov2=generated_SPD_4x4(dim=4,l=l,b=b)

csp_eigvecs, projected_eigvals1, projected_eigvals2 = basic_csp(cov1,cov2)
print (projected_eigvals1  + projected_eigvals2)
print (csp_eigvecs)
print (csp_eigvecs[:,0])
print (csp_eigvecs[:,1])






#EXAMPLE 2
l=0.9
b=0.3
c1,c2=generated_SPD_4x4(dim=4,l=l,b=b)

ts1,c1s=generate_time_series(Nsubjects=100,timepoints=500,cov=c1)
ts2,c2s=generate_time_series(Nsubjects=100,timepoints=500,cov=c2)

all_covs=np.concatenate((c1s,c2s),axis=0)
all_labels=np.concatenate([np.ones(c1s.shape[0]),np.zeros(c2s.shape[0])])
spade_acc=cv_classification_spade(all_covs=all_covs,all_labels=all_labels,Nfolds=10)

print(spade_acc)


