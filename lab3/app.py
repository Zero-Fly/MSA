from sklearn.datasets import make_moons, make_blobs
from kMeans import kMeans

class app:
    def __init__(self):
        x_moon, moon_clust = make_moons(n_samples=200, noise=0.15, random_state=42)
        self.moons = kMeans(x_moon, moon_clust, nK=6)
        wellN, wellK = 500, 4
        blobs, blobs_clust = make_blobs(n_samples=wellN, centers=wellK, cluster_std=1.0, random_state=23)
        self.wellSep = kMeans(blobs, blobs_clust, n=wellN, nK=wellK)
    
    def run(self):
        #self.moons.plotX()
        #self.moons.plot()
        #self.wellSep.plotX()
        #self.wellSep.printRes()
        self.wellSep.serial_plot()
