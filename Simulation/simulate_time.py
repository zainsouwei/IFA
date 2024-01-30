import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.decomposition import FastICA, PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
from scipy.io import savemat


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
        if len(group.shape) == 2:
            group = group[np.newaxis,:,:]
        for subjectC in group:
            n = subjectC.shape[0]
            # TODO add check to see if number of filters is less than n
            C = tds + subjectC
            # TODO Check the ordering of matrices in eigh
            _, Vsub = scipy.linalg.eigh(subjectC,b=C,subset_by_index=[n-nfilters, n-1])
            _, Vtds = scipy.linalg.eigh(tds,b=C,subset_by_index=[n-nfilters, n-1])
            filters = np.concatenate((np.abs(Vsub) ,np.abs(Vtds)),axis=1)
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
            if groupfilters.shape[0] == 1:
                subfilters = groupfilters[0,:,:]
            else:
                subfilters = groupfilters[j,:,:]
            # rpinv = np.linalg.inv(subfilters.T @ subfilters + reg*np.eye(subfilters.shape[1])) @ subfilters.T
            rpinv = subfilters.T
            timeregresed = rpinv@sub
            # trpinv = timeregresed.T@np.linalg.inv(timeregresed @ timeregresed.T + reg *np.eye(timeregresed.shape[0]))
            trpinv = timeregresed.T
            spaceregressed = trpinv.T@sub.T
            drfilters.append(spaceregressed)
            # dualregressionecheck(subfilters,timeregresed,spaceregressed,sub)
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
            if groupfilters.shape[0] == 1:
                subfilters = groupfilters[0,:,:]
            else:
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

def dualregressionecheck(subfilters,timeregresed,spaceregressed,sub):
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(subfilters[:,0:1],aspect='auto')
    plt.subplot(2,2,2)
    plt.imshow(timeregresed.T[:,0:1].T,aspect='auto')
    plt.subplot(2,2,3)
    plt.imshow(spaceregressed.T[:,0:1],aspect='auto')
    plt.subplot(2,2,4)
    plt.imshow(sub,aspect='auto')
    plt.show()   

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(subfilters[:,1:],aspect='auto')
    plt.subplot(2,2,2)
    plt.imshow(timeregresed.T[:,1:].T,aspect='auto')
    plt.subplot(2,2,3)
    plt.imshow(spaceregressed.T[:,1:],aspect='auto')
    plt.subplot(2,2,4)
    plt.imshow(sub,aspect='auto')
    plt.show()    


def PCAscree(data):
    pca = PCA(n_components=np.min(data.shape))
    pca.fit(data)
    # Plot the scree or elbow plot
    elbow = (np.sum((np.abs(np.gradient(pca.explained_variance_ratio_)) > .01)))
    print(elbow)
    plt.plot(range(1, np.min(data.shape)+1), pca.explained_variance_ratio_, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    return pca.components_[:elbow,:].T,elbow

# Define a function to run k-means clustering and visualize the results
def run_kmeans_and_visualize(data, k, title):
    # Standardize the data (optional but often recommended)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply k-means algorithm
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data_scaled)

    # Reduce dimensionality for visualization (using PCA)
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_scaled)

    # Plot the data points in 3D with color-coded clusters
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels, cmap='viridis', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    # Plot cluster centroids
    centroids = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200)

    plt.show()

if __name__ == "__main__":
    parcels = 165
    samples = 100
    ngroups = 3
    nshared = 2
    nfilters = 2
    sublevel = False
    noise = 0.3
    compfreq = 5
    # Generate Data
    allgroups, locations = makesubjects(ngroups=ngroups,nparcels=parcels,nsamples=samples,maxp=compfreq,nshared=nshared,general_noise_variance=noise)

    # TODO Demean and normalize all data
    # scaler = StandardScaler()
    # allgroups_scaled = scaler.fit_transform(allgroups)

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
    # Compare subject to group For group to group
    if sublevel:
        FCs = covs[1:]
    else:
        FCs = avgcovs[1:]

    # Begin SPADE Pipeline
    filters = spade(typical,FCs,nfilters=int(nfilters/2))
    # spadecheck(filters,locations)
    if sublevel:
        allsubsfilters = reshapefilters(filters)
    else:
        allsubsfilters = dualregression(allgroups[1:],filters,reg=1)
    
    # Clustering on Filters
    subsasfilters = np.reshape(allsubsfilters,(-1,parcels*nfilters))
    _,elbow2 = PCAscree(subsasfilters)
    run_kmeans_and_visualize(subsasfilters, elbow2, 'Cluster Filters as Features')
    
    # Clustering on Group PCA Dual Regression of Filters
    filterprincipalcomponents, elbow = PCAscree(allsubsfilters)
    PCAdualregresseddata = []
    for i in range(0, len(allsubsfilters), nfilters):
        # Select two consecutive rows
        two_rows = allsubsfilters[i:i+nfilters]
        # Perform your operations on the selected two rows
        PCAtimedualregressed = two_rows @ filterprincipalcomponents
        PCAspacedualregressed = two_rows.T @ PCAtimedualregressed
        PCAdualregresseddata.append(PCAspacedualregressed.flatten())
    PCAdualregresseddata = np.array(PCAdualregresseddata)
    run_kmeans_and_visualize(PCAdualregresseddata, elbow-1, 'Clustering on Group PCA Dual Regression of Filters')
    
    # Clustering on SubLevel ICA on Filters
    ICAsubleveldata = []
    for i in range(0, len(allsubsfilters), nfilters):
        # Select two consecutive rows
        two_rows = allsubsfilters[i:i+nfilters]
        # Perform your operations on the selected two rows
        subcomp,_ = ICA(two_rows,locations,nfilters)
        ICAsubleveldata.append(subcomp.flatten())
    ICAsubleveldata = np.array(ICAsubleveldata)
    run_kmeans_and_visualize(ICAsubleveldata, elbow-1, 'Clustering on Sublevel ICA of Filters')

    # Group ICA on Filters
    components, _ = ICA(allsubsfilters,locations,elbow)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(locations,aspect='auto')
    plt.title("Original Locations")
    plt.subplot(1,2,2)
    plt.imshow(abs(components),aspect='auto')
    plt.title("ICA Comp")
    plt.show()

    # Clustering on Group ICA Dual Regression of Filters
    GroupICAdualregresseddata = []
    for i in range(0, len(allsubsfilters), nfilters):
        # Select two consecutive rows
        two_rows = allsubsfilters[i:i+nfilters]
        # Perform your operations on the selected two rows
        GroupICAtimedualregressed = two_rows @ components
        GroupICAspacedualregressed = two_rows.T @ GroupICAtimedualregressed
        GroupICAdualregresseddata.append(GroupICAspacedualregressed.flatten())
    GroupICAdualregresseddata = np.array(GroupICAdualregresseddata)
    run_kmeans_and_visualize(GroupICAdualregresseddata, elbow-1, 'Clustering on Group ICA Dual Regression of Filters')

    if not sublevel:
        allfilters = reshapefilters(filters)
        allfiltersprincipalcomponents, allfilterselbow = PCAscree(allfilters)
        # Apply ICA to identify spatial components TODO make ICA actually blind uing PCA elbow with correct threshold
        filterscomponents, _ = ICA(allsubsfilters,locations,ngroups)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(locations,aspect='auto')
        plt.title("Original Locations")
        plt.subplot(1,2,2)
        plt.imshow(abs(filterscomponents),aspect='auto')
        plt.title("Group SPADE ICA of Group Filters")
        plt.show()

        SPADEdualregresseddata = []
        for group in allgroups:
            for sub in group[1]:
                sub = np.array(sub)
                timedualregressed = allfiltersprincipalcomponents.T@sub
                spacedualregressed = sub@timedualregressed.T
                SPADEdualregresseddata.append(spacedualregressed.flatten())
        SPADEdualregresseddata = np.array(SPADEdualregresseddata)
        run_kmeans_and_visualize(SPADEdualregresseddata, elbow-1, 'Cluster Group SPADE ICA Dual Regressed')

    # # # TODO Figure Out Baseline Dual Regression Method: Currently stuck reducing sample size
    alldata = []
    for group in allgroups:
        alldata.extend(np.concatenate(group[1], axis=1).T)
    alldata = np.array(alldata)
    # Apply ICA to identify spatial components TODO make ICA actually blind uing PCA elbow with correct threshold
    transformed, elbow3 = PCAscree(alldata.T)
    componentsdual, _ = ICA(alldata,locations,ngroups+nshared)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(locations,aspect='auto')
    plt.title("Original Locations")
    plt.subplot(1,2,2)
    plt.imshow(abs(componentsdual),aspect='auto')
    plt.title("ICA")
    plt.show()

    transformedalldata = alldata@componentsdual
    run_kmeans_and_visualize(np.reshape(transformedalldata, (-1,(ngroups+nshared)*samples)), ngroups, 'Cluster ICA Time regressed')
    
    dualregresseddata = []
    for group in allgroups:
        for sub in group[1]:
            sub = np.array(sub)
            timedualregressed = componentsdual.T@sub
            spacedualregressed = sub@timedualregressed.T
            dualregresseddata.append(spacedualregressed.flatten())
    dualregresseddata = np.array(dualregresseddata)
    run_kmeans_and_visualize(dualregresseddata, ngroups, 'Cluster ICA Dual Regressed')
