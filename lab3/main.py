from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt

# Хорошо разделенные данные
X_well_separated, y_well = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Плохо разделенные данные
X_poorly_separated, y_poor = make_moons(n_samples=300, noise=0.2, random_state=42)

plt.scatter(X_well_separated[:, 0], X_well_separated[:, 1], c=y_well, s=50, cmap='viridis')
plt.title("make_blobs example with 4 clusters")
plt.show()

plt.scatter(X_poorly_separated[:, 0], X_poorly_separated[:, 1], c=y_poor, s=50, cmap='viridis')
plt.title("make_blobs example with 4 clusters")
plt.show()