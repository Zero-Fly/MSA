#from sklearn.cluster import KMeans
from matplotlib import axis
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import numpy as np
from sympy import false

class kMeans:
    def __init__(self, X, Y, addKernel=False, n=200, gamma=0.5, nK=2):
        self.n = n
        self.gamma = gamma
        self.nK = nK
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Jc = []
        # Compute kernel matrix
        self.K = rbf_kernel(self.X, self.X, gamma=self.gamma)
        if addKernel:
            self.sK = self.simpl_kernel()
            customX = [(x[0]**3, x[1]**3) for x in X]
            self.cK = rbf_kernel(customX, customX, gamma=self.gamma)
            self.kList = [self.K, self.sK, self.cK]

    def simpl_kernel(self):
        size = self.X.shape[0]
        res = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                res[i][j] = 2*np.linalg.norm(self.X[i]) * np.linalg.norm(self.X[j])
        return res
    def cluster_central(self, clust):
        cent = np.sum(clust, axis=0)/np.size(clust)
        return cent

    def kernel_kmeans(self, nk, max_iter=1000, kernel_id=0):     
        # Random initialization
        kernel = []

        if kernel_id == 0: kernel = self.K
        elif kernel_id == 1: kernel = self.sK
        else: kernel = self.cK

        clusters = np.random.randint(0, nk, len(self.X))
        jcKey = False #only for test
        for _ in range(max_iter):
            # Assign points to clusters
            distances = np.zeros((len(self.X), nk))
            for k in range(nk):
                mask = (clusters == k)
                clust = np.take(self.X, mask)
                #print(clust)
                if np.sum(mask) == 0:
                    continue
                # Compute distance to cluster k
                distances[:, k] = (
                    np.diag(kernel) - 
                    2 * np.sum(kernel[:, mask], axis=1) / np.sum(mask) + 
                    np.sum(kernel[mask][:, mask]) / np.sum(mask)**2
                )
            
            
            new_clusters = np.argmin(distances, axis=1)
            # Check convergence
            if np.all(clusters == new_clusters):
                jc = 0
                if jcKey:
                    for k in range(nk):
                        mask = (clusters == k)
                        clust = np.take(self.X, mask)
                        cn = self.cluster_central(clust)
                        jc += sum(list(map(lambda x: (np.linalg.norm((x - cn))), clust)))
                    print(f"J(C) = {jc}, K = {nk}")
                break
            clusters = new_clusters
        
        return clusters
    
    def kernel_silhouette_score(self, labels, kernel_id=0):
        """
        Compute Silhouette scores using a precomputed kernel matrix.
        
        Parameters:
        - K: Kernel matrix (n_samples, n_samples)
        - labels: Cluster labels (n_samples,)
        
        Returns:
        - s: Average Silhouette score
        """
        kernel = []

        if kernel_id == 0: kernel = self.K
        elif kernel_id == 1: kernel = self.sK
        else: kernel = self.cK

        n_samples = kernel.shape[0]
        unique_labels = np.unique(labels)
        s = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Intra-cluster distance (a_i)
            mask_same = (labels == labels[i])
            a_i = kernel[i, i] - 2 * np.mean(kernel[i, mask_same]) + np.mean(kernel[mask_same][:, mask_same])
            
            # Nearest-cluster distance (b_i)
            b_i = np.inf
            for k in unique_labels:
                if k == labels[i]:
                    continue
                mask_other = (labels == k)
                dist = kernel[i, i] - 2 * np.mean(kernel[i, mask_other]) + np.mean(kernel[mask_other][:, mask_other])
                if dist < b_i:
                    b_i = dist
            
            # Silhouette score
            if max(a_i, b_i) != 0:
                s[i] = (b_i - a_i) / max(a_i, b_i) 
            else:
                s[i] = 0
        
        return np.mean(s)
    
    def printRes(self, maxK=0, kernel_id=0):
        if maxK == 0:
            maxK = self.nK + 4
        for i in range(2, maxK):
            clusters = self.kernel_kmeans(i, kernel_id=kernel_id)
            print('| K = ' + str(i) + ',   Sil = ', end='')
            print(f'{self.kernel_silhouette_score(clusters, kernel_id=kernel_id):.3}', end=' |\n')

    def serial_plot(self, maxK=0, kernel_id=0):
        if maxK == 0:
            maxK = self.nK + 4
        fig = plt.figure(figsize=(10, 12))
        subplt = [fig.add_subplot(3,2,i+1) for i in range(maxK-2)]
        for i in range(2, maxK):
            clusters = self.kernel_kmeans(i, kernel_id=kernel_id)
            subplt[i-2].scatter(self.X[:, 0], self.X[:, 1], c=clusters, cmap='viridis')
            subplt[i-2].set_title(f"{i} clusters")
        plt.show()

    def plot(self):
        clusters = self.kernel_kmeans(self.nK)
        # Plot clusters
        plt.scatter(self.X[:, 0], self.X[:, 1], c=clusters, cmap='viridis')
        plt.title("Kernel K-Means")
        plt.show()

    def plotX(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap='viridis')
        plt.title("Input data")
        plt.show()

    def plot_compare(self, nK):
        fig = plt.figure()
        clusters = self.kernel_kmeans(nK)
        # Plot clusters
        kmean = fig.add_subplot(1,2,1)
        kmean.scatter(self.X[:, 0], self.X[:, 1], c=clusters, cmap='viridis')
        kmean.set_title("Kernel K-Means")

        input_clust = fig.add_subplot(1,2,2)
        input_clust.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap='viridis')
        input_clust.set_title("Input data")

        plt.show()

    def serial_kernal(self, nK):
        fig = plt.figure(figsize=(16, 5))
        rbf_cluster = self.kernel_kmeans(nK)
        simple_cluster = self.kernel_kmeans(nK, kernel_id=1)
        custom_cluster = self.kernel_kmeans(nK, kernel_id=2)
        # Plot clusters
        rbf_kmean = fig.add_subplot(1,3,1)
        rbf_kmean.scatter(self.X[:, 0], self.X[:, 1], c=rbf_cluster, cmap='viridis')
        rbf_kmean.set_title("RBF Kernel")

        simple_kmean = fig.add_subplot(1,3,2)
        simple_kmean.scatter(self.X[:, 0], self.X[:, 1], c=simple_cluster, cmap='viridis')
        simple_kmean.set_title("Trivial Kernel")

        custom_kmean = fig.add_subplot(1,3,3)
        custom_kmean.scatter(self.X[:, 0], self.X[:, 1], c=custom_cluster, cmap='viridis')
        custom_kmean.set_title("Custom Kernel")

        plt.show()