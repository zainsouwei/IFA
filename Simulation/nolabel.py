import os 
import sys
import numpy as np
#import matplotlib.pyplot as plt
import copy as cp
import time
from scipy.linalg import eigh
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import matplotlib.pyplot as plt
#modify next line to provide the  path to your spade directory
toolbox_path = r"C:\Users\zaisou\Desktop\spade_zain\spade_tools_python"





sys.path.append(os.path.abspath(toolbox_path)) 
# from spade_tools_python import spade_tools, generate_spade_toys
from spade_tools import basic_csp,cv_classification_spade
from generate_spade_toys import generated_SPD_4x4, generate_time_series

b=0.3
# c1,c2=generated_SPD_4x4(dim=4,l=0.86,b=b)
# c3,c4=generated_SPD_4x4(dim=4,l=.7,b=b)
u1 = np.array([0, 5, 0])
u2 = np.array([0, -5, 0])
# c1 = .5*np.array([[8,0,0],[0,26,0],[0,0,8]])
# c2 = .5*np.array([[2,0,0],[0,26,0],[0,0,2]])
c1 = np.array([[8,0,0],[0,1,0],[0,0,8]])
c2 = np.array([[2,0,0],[0,1,0],[0,0,2]])
r1 = .5*c1+u1@u1.T
r2 = .5*c2+u2@u2.T
print(c1)
print(c2)
w, v = eigh(r1, r1+r2, eigvals_only=False)
print(v.T@v)

# Generate data points for plotting
num_points = 100
data1 = np.random.multivariate_normal(u1, c1, num_points)
data2 = np.random.multivariate_normal(u2, c2, num_points)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], label='Class 1', alpha=0.7)
ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], label='Class 2', alpha=0.7)

# Plot eigenvectors
origin = np.zeros(3)
scale_factor = 5  # Adjust the scale factor for better visualization
for i in range(len(w)):
    ax.quiver(*origin, *v[:, i] * w[i] * scale_factor, color='red', label=f'Eigenvector {i + 1}', arrow_length_ratio=0.1)

# Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Dataset and Eigenvectors')

# Add legend
ax.legend()

# Show the plot
plt.show()



ts1,c1s=generate_time_series(Nsubjects=100,timepoints=500,cov=c1)
ts2,c2s=generate_time_series(Nsubjects=100,timepoints=500,cov=c2)
# ts3,c3s=generate_time_series(Nsubjects=100,timepoints=500,cov=c3)
# ts4,c4s=generate_time_series(Nsubjects=100,timepoints=500,cov=c4)

theeohCs = []
theeohCs.extend(c1s)
theeohCs.extend(c2s)
# theeohCs.extend(c3s)
# theeohCs.extend(c4s)
theeohCs = np.array(theeohCs)

all = []
for oh in theeohCs:
    row = []
    for Cs in theeohCs:
        eigs = eigh(oh, oh+Cs, eigvals_only=True)
        row.append(np.sum(abs(eigs[0]-.5)+abs(eigs[3]-.5)))
        # row.append(np.sum(abs(eigh(oh, oh+Cs, eigvals_only=True)-.5)))
        # row.append(np.var(eigh(oh, oh+Cs, eigvals_only=True)))
    all.append(row)
all = np.array(all)



# Use MDS to reduce dimensionality to 2D
mds = MDS(n_components=2, dissimilarity='precomputed')
embedding = mds.fit_transform(all)

# Label the points and visualize
plt.scatter(embedding[:100, 0], embedding[:100, 1], label='Class 1', c='blue')
plt.scatter(embedding[100:200, 0], embedding[100:200, 1], label='Class 2', c='red')
# plt.scatter(embedding[200:300, 0], embedding[200:300, 1], label='Class 3', c='green')
# plt.scatter(embedding[300:400, 0], embedding[300:400, 1], label='Class 4', c='yellow')

# Customize the plot
plt.title('MDS Visualization with Class Labels')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.legend()
plt.show()


## See how it compares with nondiscriminatory MDS
# Reshape theeohCs to 2D (200x16)
reshaped_theeohCs = theeohCs.reshape((200, -1))

# Use MDS to reduce dimensionality to 2D
nondiscrimMDS = MDS(n_components=2)
nondiscrimembedding = nondiscrimMDS.fit_transform(reshaped_theeohCs)

# Visualize the points
plt.scatter(nondiscrimembedding[:100, 0], nondiscrimembedding[:100, 1], label='Class 1', c='blue')
plt.scatter(nondiscrimembedding[100:, 0], nondiscrimembedding[100:, 1], label='Class 2', c='red')

# Customize the plot
plt.title('MDS no discrim Visualization with Class Labels')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.legend()
plt.show()