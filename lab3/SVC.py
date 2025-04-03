import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.datasets import make_blobs, make_moons

Q = 0.1
P = 0.05
# 1. Генерация данных (концентрические окружности)
np.random.seed(42)
n_samples = 200
theta = np.linspace(0, 2*np.pi, n_samples)
r1 = 1.5*np.random.rand(n_samples) + 0.5
r2 = 3.0*np.random.rand(n_samples) + 2.0

# X = np.vstack([
#     np.column_stack([r1*np.cos(theta), r1*np.sin(theta)]),
#     np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])
# ])

# Плохо разделенные данные
X, y_poor = make_blobs(n_samples=200, centers=3, cluster_std=1.4, random_state=29)

# 2. Функция RBF-ядра
def rbf_kernel(X, gamma=0.5):
    """Вычисляет матрицу RBF-ядра"""
    sq_dists = np.sum(X**2, axis=1).reshape(-1,1) + np.sum(X**2, axis=1) - 2*np.dot(X, X.T)
    return np.exp(-gamma * sq_dists)

# 3. Решение двойственной задачи в формулировке Вольфа
def wolf_dual_sphere_clustering(X, nu=0.1, gamma=0.5):
    n = len(X)
    K = rbf_kernel(X, gamma)
    
    # Параметры для cvxopt
    P = matrix(K)
    q = matrix(-np.diag(K))
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = matrix(np.hstack([np.zeros(n), np.ones(n)*nu]))
    A = matrix(np.ones((1,n)))
    b = matrix(1.0)
    
    # Решение квадратичной задачи
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x']).flatten()
    
    # Вычисление радиуса R
    sv_indices = (alphas > 1e-5)
    center_dists = np.diag(K) - 2*np.dot(K, alphas) + np.dot(alphas.T, np.dot(K, alphas))
    R = np.sqrt(np.median(center_dists[sv_indices]))
    
    return alphas, R, sv_indices

# 4. Применение алгоритма
alphas, R, sv_indices = wolf_dual_sphere_clustering(X, nu=P, gamma=Q)

# 5. Вычисление расстояний до центра
K = rbf_kernel(X)
center_dists = np.diag(K) - 2*np.dot(K, alphas) + np.dot(alphas.T, np.dot(K, alphas))

# 6. Кластеризация (внутри/вне гиперсферы)
labels = np.where(center_dists <= R, 1, -1)

# 7. Визуализация
plt.figure(figsize=(15, 5))

# Исходные данные с опорными векторами
plt.subplot(131)
plt.scatter(X[:,0], X[:,1], c='blue', alpha=0.5)
plt.scatter(X[sv_indices,0], X[sv_indices,1], 
            s=50, facecolors='none', edgecolors='red',
            label='Опорные векторы')
plt.title(f"Опорные векторы (nu={0.1})")
plt.legend()

# Расстояния до центра
plt.subplot(132)
plt.scatter(X[:,0], X[:,1], c=center_dists, cmap='viridis')
plt.colorbar(label='Расстояние до центра')
plt.title("Расстояние до центра в пространстве признаков")

# Результат кластеризации
plt.subplot(133)
plt.scatter(X[:,0], X[:,1], c=labels, cmap='coolwarm')
plt.title("Кластеры (внутри/вне гиперсферы)")

plt.tight_layout()
plt.show()


#---------------------------
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