#from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import numpy as np

class kMeans:
    def __init__(self, X, Y, n=200, gamma=0.5, nK=2):
        self.n = n
        self.gamma = gamma
        self.nK = nK
        self.X = X
        self.Y = Y
        # Compute kernel matrix
        self.K = rbf_kernel(self.X, self.X, gamma=self.gamma)

    def kernel_kmeans(self, nk, max_iter=1000):     
        # Random initialization
        clusters = np.random.randint(0, nk, len(self.X))
        
        for _ in range(max_iter):
            # Assign points to clusters
            distances = np.zeros((len(self.X), nk))
            for k in range(nk):
                mask = (clusters == k)
                if np.sum(mask) == 0:
                    continue
                # Compute distance to cluster k
                distances[:, k] = (
                    np.diag(self.K) - 
                    2 * np.sum(self.K[:, mask], axis=1) / np.sum(mask) + 
                    np.sum(self.K[mask][:, mask]) / np.sum(mask)**2
                )
            new_clusters = np.argmin(distances, axis=1)
            
            # Check convergence
            if np.all(clusters == new_clusters):
                break
            clusters = new_clusters
        
        return clusters
    
    def kernel_silhouette_score(self, labels):
        """
        Compute Silhouette scores using a precomputed kernel matrix.
        
        Parameters:
        - K: Kernel matrix (n_samples, n_samples)
        - labels: Cluster labels (n_samples,)
        
        Returns:
        - s: Average Silhouette score
        """
        n_samples = self.K.shape[0]
        unique_labels = np.unique(labels)
        s = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Intra-cluster distance (a_i)
            mask_same = (labels == labels[i])
            a_i = self.K[i, i] - 2 * np.mean(self.K[i, mask_same]) + np.mean(self.K[mask_same][:, mask_same])
            
            # Nearest-cluster distance (b_i)
            b_i = np.inf
            for k in unique_labels:
                if k == labels[i]:
                    continue
                mask_other = (labels == k)
                dist = self.K[i, i] - 2 * np.mean(self.K[i, mask_other]) + np.mean(self.K[mask_other][:, mask_other])
                if dist < b_i:
                    b_i = dist
            
            # Silhouette score
            s[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        
        return np.mean(s)
    
    def printRes(self, maxK=0):
        if maxK == 0:
            maxK = self.nK + 3
        print('|', end='')
        for i in range(2, maxK):
            clusters = self.kernel_kmeans(i)
            print(' i = ' + str(i) + ', s = ', end='')
            print(f'{self.kernel_silhouette_score(clusters):.3}', end=' |')

    def serial_plot(self, maxK=0):
        fig = plt.figure(figsize=(10, 12))
        if maxK == 0:
            maxK = self.nK + 4
        for i in range(2, maxK):
            clusters = self.kernel_kmeans(i)
            plt.scatter(self.X[:, 0], self.X[:, 1], c=clusters, cmap='viridis')
            plt.title("Kernel K-Means")
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