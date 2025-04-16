from networkx import sigma
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


class LinReg:
    def __init__(self, fX="res/X.txt", fY="res/Y.txt", test=True):
        self.X, self.Y = self.readFile(fX, fY)
        self.sigma = 0.01
        self.std = self.sigma**0.5
        self.eps = self.addEps(std=self.std)
        self.XT = self.X.T
        self.XTX = self.XT @ self.X
        self.XTX_inv = np.linalg.inv(self.XTX)


        if test:
            print(self.X)
            print(self.Y)
            print('---------------')
            self.SSE()

    def addEps(self, std=0.1, size=15, seed=52):
        np.random.seed(seed)
        eps = np.random.normal(loc=0, scale=std, size=size)
        self.X[:, -1] += eps
        return eps

    def SSE(self):
        
        a = self.XTX_inv @ self.XT @ self.Y
        a_size = np.size(a)
        print(f'a: {a}\n')
        y_ = self.X @ a
        print(f'y^: \n{y_}\n')
        e = self.Y - y_
        print(f'e: \n{e}\n')
        Me = np.mean(e)
        print(f'M[e] = {Me}\n')
        ss = e.T @ e / 11
        print(f's² : {ss}\n')
        cov_a = ss * self.XTX_inv
        print(f'cov^(a) :\n{cov_a}\n')
        std_ = np.array([cov_a[i,i]**0.5 for i in range(a_size)])
        print(f's(a_i) : {std_}\n')
        cor_a = np.array(cov_a)
        for i in range(a_size):
            for j in range(a_size):
                cor_a[i, j] /= (cov_a[i,i] * cov_a[j,j])**0.5
        print(f'cor(a) : \n{cor_a}\n')
        #self.print_hist(e, bins=3)
        R_2 = 1 - np.sum(e**2)/sum((self.Y - np.mean(self.Y))**2)
        R_2n = 1 - np.sum(e**2)/sum((self.Y - np.mean(self.Y))**2) * (14/11)
        print(f'R² = {R_2},   R²_н = {R_2n}\n')
        self.print_Di(a, std_)

        for i in range(self.Y):
            rmX = np.delete(self.X, i)
            rmY = np.delete(self.Y, i)
            


    def readFile(self, fX, fY):
        X = np.loadtxt(fX, dtype=float)
        Y = np.loadtxt(fY, dtype=float)
        return X, Y
    
    def printSSE(self):
        a = self.XTX_inv @ self.XT @ self.Y
        print(a)

    def print_Di(self, a, s, alpha=0.05):
        t_critical = stats.t.ppf(1 - alpha/2, len(self.Y) - self.X.shape[1])
        ci_lower = a - t_critical * s
        ci_upper = a + t_critical * s

        print(f"95% ДИ:\n")
        for i in range(self.X.shape[1]):
            check = 'ERROR'
            if a[i] > ci_lower[i] and a[i]<ci_upper[i]:
                check = 'OK'
            print(f'[ {ci_lower[i]:.4}  {ci_upper[i]:.4}]  {check} (a_i)', end="\n")
        print()


    def print_hist(self, data, bins=5, norm=True):
        # Гистограмма остатков
        plt.figure(figsize=(10, 6))
        #sns.histplot(residuals, kde=True, bins=5, color='blue', stat='density')
        plt.hist(data, bins=bins, density=norm, alpha=0.7, color='blue', edgecolor='black', label='Нормированная гистограмма')
        plt.title('Гистограмма остатков с KDE')
        plt.xlabel('Остатки (y - ŷ)')
        plt.ylabel('Плотность')
        plt.axvline(x=0, color='red', linestyle='--')  # Нулевая линия
        plt.show()

    def MSE(self):
        pass