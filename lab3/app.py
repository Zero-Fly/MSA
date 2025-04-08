from sklearn.datasets import make_moons, make_blobs, make_circles
from kMeans import kMeans

class app:
    def __init__(self):
        x_circle, y_circle = make_circles(200, noise=0.05, random_state=24, factor=0.6)
        self.circle = kMeans(x_circle, y_circle, n=401, nK=2, addKernel=True)

        x_moon, moon_clust = make_moons(n_samples=200, noise=0.11, random_state=42)
        self.moons = kMeans(x_moon, moon_clust, nK=4, addKernel=True)

        wellN, wellK = 500, 4
        blobs, blobs_clust = make_blobs(n_samples=wellN, centers=wellK, cluster_std=1.0, random_state=23)
        self.wellSep = kMeans(blobs, blobs_clust, n=wellN, nK=wellK)
        
        blobs_2, blobs_clust_2 = make_blobs(n_samples=wellN, centers=wellK, cluster_std=1.6, random_state=32)
        self.blob2 = kMeans(blobs_2, blobs_clust_2, n=wellN, nK=wellK)
    
    def run(self):
        #self.circle.plotX()
        #self.circle.plot()
        #self.circle.serial_plot()
        #self.circle.printRes()
        #self.circle.serial_kernal(5)

        #self.blob2.plotX()
        #self.blob2.plot()
        #self.blob2.serial_plot()
        #self.blob2.printRes()
        #self.blob2.plot_compare(4)

        #self.moons.plotX()
        #self.moons.plot()
        self.moons.serial_plot(kernel_id=2)
        #self.moons.printRes(kernel_id=1)
        #self.moons.serial_kernal(3)

        #self.wellSep.plot()
        #self.wellSep.plotX()
        #self.wellSep.printRes()
        #self.wellSep.serial_plot()
        #self.wellSep.plot_compare(4)
        pass