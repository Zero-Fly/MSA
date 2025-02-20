import numpy as np
import scipy.stats as stats
from scipy.stats import skew, kurtosis, zscore, normaltest, shapiro, norm
import matplotlib.pyplot as plt

class MPC:
    def __init__(self):
        self.data = np.array([])
        self.filtDate = np.array([])
        self.stat = {
            'mean':0,
            'var' :0,
            'skew':0,
            'kurt':0
        }
        self.BINS = 5

    def run(self):
        self.readFile("resorces/Number_28.txt")
        self.getStat()
        self.plotHnF()
        self.confBands()
        self.resGraph()
        self.qqTest(self.data)
        self.filteredData()
        self.resGraph()
        self.qqTest(self.data)
        

    def readFile(self, filename):
        # Open the file in read mode
        with open(filename, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()
            
            # Initialize an empty list to store all float numbers
            float_numbers = np.array([])
            
            # Iterate over each line
            for line in lines:
                # Split the line into individual strings based on spaces
                numbers = line.strip().split()
                
                # Convert each string to a float and add to the list
                for num in numbers:
                    try:
                        float_num = float(num)
                        float_numbers = np.append(float_numbers, float_num)
                    except ValueError:
                        print(f"Skipping invalid float: {num}")
            self.data = float_numbers
            # Print the list of float numbers
            print("All float numbers in the file:\n", float_numbers)

    def printStat(self):
        print(f"Выборочное среднее: {self.stat['mean']}\n" +
              f"Выборочная дисперсия: {self.stat['var']}\n" +
              f"Коэффициент асимметрии: {self.stat['skew']}\n" +
              f"Коэффициент эксцесса: {self.stat['kurt']}")

    def getStat(self):
        # Выборочное среднее
        self.stat['mean'] = np.mean(self.data)
        # Выборочная дисперсия
        self.stat['var'] = np.var(self.data)
        # Выборочный коэффициент асимметрии
        self.stat['skew'] = skew(self.data)
        # Выборочный коэффициент эксцесса (нормированный, с вычетом 3)
        self.stat['kurt'] = kurtosis(self.data)
        self.printStat()

    def empiricalCDF(self, data):
        n = len(data)
        x = np.sort(data)
        y = np.arange(1, n + 1) / n
        return x, y
    
    # Построение нормированной гистограммы
    def plotNormedHistogram(self, data, bins=5):
        plt.hist(data, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black', label='Нормированная гистограмма')
        plt.xlabel('Значения')
        plt.ylabel('Плотность')
        plt.title('Нормированная гистограмма')
        plt.legend()
        plt.grid(True)

    # Построение ЭФР
    def plotEcdf(self, x_ecdf, y_ecdf):
        plt.step(x_ecdf, y_ecdf, where='post', color='red', label='ЭФР')
        plt.xlabel('Значения')
        plt.ylabel('F(x)')
        plt.title('Эмпирическая функция распределения')
        plt.legend()
        plt.grid(True)

    def plotHnF(self):
        # Сортировка данных для построения ЭФР
        x_ecdf, y_ecdf = self.empiricalCDF(self.data)
        # Создание графиков
        plt.figure(figsize=(12, 6))

        # График 1: Нормированная гистограмма
        plt.subplot(1, 2, 1)
        self.plotNormedHistogram(self.data)

        # График 2: ЭФР
        plt.subplot(1, 2, 2)
        self.plotEcdf(x_ecdf, y_ecdf)

        plt.tight_layout()
        plt.show()

    def filteredData(self):
        #z_scores = zscore(self.data)

        # Фильтрация выбросов
        #self.filtDate = self.data[(z_scores < 3) & (z_scores > -3)]
        for num in self.data:
            if num < 3.5:
                self.filtDate = np.append(self.filtDate, num)
        #print(self.filtDate)
        plt.figure(figsize=(6, 6))
        self.plotNormedHistogram(self.filtDate)
        # Выборочное среднее
        self.stat['mean'] = np.mean(self.filtDate)
        # Выборочная дисперсия
        self.stat['var'] = np.var(self.filtDate)
        # Выборочный коэффициент асимметрии
        self.stat['skew'] = skew(self.filtDate)
        # Выборочный коэффициент эксцесса (нормированный, с вычетом 3)
        self.stat['kurt'] = kurtosis(self.filtDate)
        self.printStat()

    def confBands(self):
        n = self.data.size
        x_ecdf, y_ecdf = self.empiricalCDF(self.data)

        # Критические значения для доверительных полос
        d_90 = 1.22 / np.sqrt(n)  # Для доверительной вероятности 0.90
        d_95 = 1.36 / np.sqrt(n)  # Для доверительной вероятности 0.95

        # Построение доверительных полос
        y_lower_90 = np.maximum(y_ecdf - d_90, 0)  # Нижняя граница (0.90)
        y_upper_90 = np.minimum(y_ecdf + d_90, 1)  # Верхняя граница (0.90)
        y_lower_95 = np.maximum(y_ecdf - d_95, 0)  # Нижняя граница (0.95)
        y_upper_95 = np.minimum(y_ecdf + d_95, 1)  # Верхняя граница (0.95)

        # Построение графика
        plt.figure(figsize=(8, 6))

        # ЭФР
        plt.step(x_ecdf, y_ecdf, where='post', color='blue', label='ЭФР')

        # Доверительные полосы (0.90)
        plt.fill_between(x_ecdf, y_lower_90, y_upper_90, color='lightblue', alpha=0.5, label='Доверительная полоса (0.90)')

        # Доверительные полосы (0.95)
        plt.fill_between(x_ecdf, y_lower_95, y_upper_95, color='lightgreen', alpha=0.3, label='Доверительная полоса (0.95)')

        # Настройка графика
        plt.xlabel('Значения')
        plt.ylabel('F(x)')
        plt.title('ЭФР с доверительными полосами')
        plt.legend()
        plt.grid(True)
        plt.show()

    def qqTest(self, data):
        # Визуальный анализ: гистограмма
        plt.figure(figsize=(12, 6))

        # Гистограмма
        plt.subplot(1, 2, 1)
        self.plotNormedHistogram(data, 7)

        # Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title('Q-Q plot')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Статистические тесты для проверки нормальности
        # Тест Шапиро-Уилка
        shapiro_test = shapiro(data)
        print(f"Тест Шапиро-Уилка: статистика = {shapiro_test.statistic}, p-value = {shapiro_test.pvalue}")

        # Тест на нормальность (D'Agostino and Pearson's test)
        norm_test = normaltest(data)
        print(f"Тест на нормальность: статистика = {norm_test.statistic}, p-value = {norm_test.pvalue}")

    def resGraph(self):
        # Построение графика
        plt.figure(figsize=(10, 6))

        # Гистограмма данных
        self.plotNormedHistogram(self.data, 7)

        # Теоретическая плотность распределения
        x = np.linspace(min(self.data) - 1, max(self.data) + 1, 1000)  # Диапазон значений для построения графика
        pdf = norm.pdf(x, self.stat['mean'], self.stat['var'])  # Плотность распределения
        plt.plot(x, pdf, 'r-', linewidth=2, label=f'Нормальное распределение (μ={self.stat['mean']:.4f}, σ={self.stat['var']:.4f})')

        # Настройка графика
        plt.xlabel('Значения')
        plt.ylabel('Плотность')
        plt.title('Гистограмма данных и теоретическая плотность распределения')
        plt.legend()
        plt.grid(True)
        plt.show()
