import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.decomposition import FastICA, PCA
import sys

def indeplocations(ngroups,nparcels,maxp=5):
    if maxp*ngroups > nparcels:
        sys.exit("Error: The condition maxp*ngroups <= nparcels is not met. Exiting program.")
    locations = []
    indices = np.random.choice(nparcels, size=maxp*ngroups, replace=False)
    for i in range(ngroups):
        array = np.full(nparcels, False)
        index = indices[i*maxp:(i+1)*maxp]
        array[index] = True
        locations.append(array.astype(int))
    locations = np.array(locations).T
    return locations

def makesubjects(ngroups=np.random.randint(2,10), nparcels=165,nsamples=600, maxp=5, nshared = 2, general_noise_variance = 0.1):
    # TODO parallelize and remove for loops
    # Generate independnet locations for all components across all subjects
    locations = indeplocations(ngroups+nshared,nparcels,maxp)

    # Create the components that all groups/subjects share
    # TODO remove for loop by using masking with columns
    allgroup_signals = np.zeros((nparcels,nsamples))
    for i in range(nshared):
        allgroup_signals += (np.random.rand(1, nsamples) - 0.5)*locations[:,i,np.newaxis]

    groups = []
    for i in range(ngroups):
        group_structure = (np.random.rand(1, nsamples) - 0.5)*locations[:,nshared+i,np.newaxis] + allgroup_signals
        nsubjects = np.random.randint(30,101)
        print("Subjects per group: ", nsubjects)
        group_data = group_structure + (general_noise_variance * np.random.randn(nsubjects,nparcels, nsamples))
        groups.append([group_structure, group_data])
    return groups, locations

def regcov(groupdata,gamma=0.1,beta=0.1,sc=1.0):
    # TODO parallelize and remove for loops
    groupC = []
    for group in groupdata:
        # Initialize arrays to store the regularized covariance matrices for each subject
        groupregC = []
        # Compute the ground truth covariance matrices for each group (before adding noise)
        groundtruth = np.array(group[0])
        groundtruthC = groundtruth@groundtruth.T
        for sub in group[1]:
            # Compute initial spatial covariance matrices for each group
            subdata = np.array(sub)
            subC = subdata@subdata.T
            I = np.identity(subC.shape[0])  # Identity matrix of the same size as subC
            # Apply the second regularization term using the ground truth covariance matrices
            Chat = (1 - beta) * sc * subC + beta * groundtruthC
            # Apply regularization to the initial covariance matrices
            Creg = (1 - gamma) * Chat + gamma * I
            groupregC.append(Creg)
        groupC.append(groupregC)
    return groupC

def spade(tds,ads,nfilters=1):
    n = tds.shape[0]
    allfilters = []
    for group in ads:
        group = np.array(group)
        groupfilters = []
        for subjectC in group:
            n = subjectC.shape[0]
            # TODO add check to see if number of filters is less than n
            C = tds + subjectC
            # TODO Check the ordering of matrices in eigh
            _, Vsub = scipy.linalg.eigh(subjectC,b=C,subset_by_index=[n-nfilters, n-1])
            _, Vtds = scipy.linalg.eigh(tds,b=C,subset_by_index=[n-nfilters, n-1])
            filters = np.concatenate((Vsub ,Vtds),axis=1)
            groupfilters.append(filters)
        allfilters.append(groupfilters)
    return allfilters

def dualregression(rawdata, filters,reg=10):
    # TODO make more efficient, dont hardcode regularization use gradient descent
    # TODO check grouplevel case, do not need to recompute pinv everytime
    drfilters = []
    for i, group in enumerate(rawdata):
        group = np.array(group[1])
        groupfilters = np.array(filters[i])
        for j, sub in enumerate(group):
            subfilters = groupfilters[j,:,:]
            # rpinv = np.linalg.inv(subfilters.T @ subfilters + reg*np.eye(subfilters.shape[1])) @ subfilters.T
            rpinv = subfilters.T
            timeregresed = rpinv@sub
            # trpinv = timeregresed.T@np.linalg.inv(timeregresed @ timeregresed.T + reg *np.eye(timeregresed.shape[0]))
            trpinv = timeregresed.T
            spaceregressed = trpinv.T@sub.T
            drfilters.append(spaceregressed)
            if j == 0:
                dualregressionecheck(subfilters,timeregresed, spaceregressed,sub)
    drfilters = np.array(drfilters)
    # TODO is this the right concatenation
    drfilters = drfilters.reshape(-1,drfilters.shape[2])
    return drfilters

def reshapefilters(filters):
    allsubsfilters = []
    for group in filters:
        for sub in group:
            allsubsfilters.append(np.array(sub).T)
    allsubsfilters = np.array(allsubsfilters)
    allsubsfilters = np.reshape(allsubsfilters,(-1,allsubsfilters.shape[2]))
    return allsubsfilters

def projection(rawdata, filters,reg=1):
    # TODO make more efficient, dont hardcode regularization use gradient descent
        # TODO check grouplevel case, do not need to recompute pinv everytime
    drfilters = []
    for i, group in enumerate(rawdata):
        group = np.array(group[1])
        groupfilters = np.array(filters[i])
        for j, sub in enumerate(group):
            subfilters = groupfilters[j,:,:]
            proj = subfilters@np.linalg.inv(subfilters.T @ subfilters + reg*np.eye(subfilters.shape[1])) @ subfilters.T
            projected = proj@sub
            drfilters.append(projected.T)
            # dualregressionecheck(subfilters,spaceregressed,sub)
    drfilters = np.array(drfilters)
    # TODO is this the right concatenation
    drfilters = drfilters.reshape(-1,drfilters.shape[2])
    return drfilters

def ICA(data,locations,ngroups):
    # Perform PCA
    # pca = PCA(n_components=min(parcels, samples))
    # pca.fit(data.T)

    # # Plot the explained variance to find the elbow
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of components')
    # plt.ylabel('Cumulative explained variance')
    # plt.show()

    # # Determine the number of components for ICA
    # n_pca_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.95)
    # print("Number of PCA components: ", n_pca_components)

    # Now use this number to specify the number of components for ICA
    ica = FastICA(n_components=ngroups)
    ica_components = ica.fit_transform(data.T)
    mixing = ica.mixing_

    # Sort Components Based on Euclidean Similarity
    component_location_mapping = []
    for i in range(ngroups):
        similarity = np.abs(ica_components[:, i]).T@locations
        component_location_mapping.append(np.argmax(similarity))
    # Get the indices that would sort the component_location_mapping array
    sorted_indices = np.argsort(component_location_mapping)
    # Reorder the components based on the sorted indices
    sorted_components = ica_components[:, sorted_indices]
    # proj = np.linalg.pinv(mixing)
    # proj = proj.T@proj
    # Oproj = np.eye(proj.shape[0]) - proj
    # print(Oproj.shape)
    return sorted_components, mixing

def covcheck(avgcovs):
    for group in avgcovs:
        plt.figure()
        plt.imshow(group)
        plt.show()

def spadecheck(filters,locations):
    # Set the colormap
    cmap = 'viridis'
    for group in filters:
        group = np.array(group)
        # First Group
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(group[0, :, :], aspect='auto', cmap=cmap)
        plt.subplot(1, 2, 2)
        plt.imshow(locations, aspect='auto', cmap=cmap)
        plt.show()

def dualregressionecheck(subfilters,timeregressed, spaceregressed,sub):
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(subfilters[:,0:1],aspect='auto')
    plt.subplot(1,3,2)
    plt.imshow(spaceregressed.T[:,0:1],aspect='auto')
    plt.subplot(1,3,3)
    plt.imshow(sub,aspect='auto')
    plt.show() 

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(timeregressed,aspect='auto')
    plt.subplot(1,3,2)
    plt.imshow(spaceregressed.T,aspect='auto')
    plt.subplot(1,3,3)
    plt.imshow(sub,aspect='auto')
    plt.show() 

    
if __name__ == "__main__":
    parcels = 165
    samples = 100
    ngroups = 3
    nshared = 2
    # Generate Data
    allgroups, locations = makesubjects(ngroups=ngroups,nparcels=parcels,nsamples=samples,maxp=5,nshared=nshared,general_noise_variance=0.1)

    # TODO Demean and normalize all data
    
    # Compute Regularized Covariance Matrices
    covs = regcov(allgroups,gamma=0.0000001,beta=0)
    avgcovs = []
    for group in covs:
        group = np.array(group)
        avgcovs.append(np.mean(group,axis=0))
    avgcovs = np.array(avgcovs)
    # covcheck(avgcovs)

    # Arbitrarily pick the first group to be the typical group
    typical = avgcovs[0]
    # Compare subject to group or group to group
    # sublevel = True
    # Begin SPADE Pipeline
    filters = spade(typical,covs[1:],nfilters=1)
    spadecheck(filters,locations)

    allsubsfiltersDR = dualregression(allgroups[1:],filters,reg=100)
    components, _ = ICA(allsubsfiltersDR,locations,ngroups)
    allsubsfiltersFilters = reshapefilters(filters)
    components1, _ = ICA(allsubsfiltersFilters,locations,ngroups)
    allsubfiltersProj = projection(allgroups[1:],filters,reg=0)
    components2, _ = ICA(allsubfiltersProj,locations,ngroups)

    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(locations,aspect='auto')
    plt.title("Original Locations")
    plt.subplot(1,4,2)
    plt.imshow(abs(components),aspect='auto')
    plt.title("Dual Regression")
    plt.subplot(1,4,3)
    plt.imshow(abs(components1),aspect='auto')
    plt.title("Filters")
    plt.subplot(1,4,4)
    plt.imshow(abs(components2),aspect='auto')
    plt.title("Projection")
    plt.show()