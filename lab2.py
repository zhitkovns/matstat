import numpy as np
import csv
import os

# Устанавливаем seed для воспроизводимости
np.random.seed(42)

# Создаём папку для результатов, если её нет
os.makedirs("results", exist_ok=True)

def compute_statistics(sample):
    """
    Вычисляет пять характеристик положения для одной выборки.
    Возвращает кортеж: (среднее, медиана, zR, zQ, усечённое среднее)
    """
    n = len(sample)
    mean = np.mean(sample)
    median = np.median(sample)
    zR = (np.min(sample) + np.max(sample)) / 2.0
    q1, q3 = np.percentile(sample, [25, 75])
    zQ = (q1 + q3) / 2.0
    # Усечённое среднее: отбрасываем по 10% с каждого конца
    trim = int(0.1 * n)
    sorted_sample = np.sort(sample)
    if trim > 0:
        trimmed = sorted_sample[trim:-trim]
    else:
        trimmed = sorted_sample
    zTr = np.mean(trimmed)
    return mean, median, zR, zQ, zTr

# Параметры эксперимента
distributions = [
    {"name": "Нормальное", "gen": lambda s: np.random.normal(0, 1, s)},
    {"name": "Коши",       "gen": lambda s: np.random.standard_cauchy(s)},
    {"name": "Лапласа",    "gen": lambda s: np.random.laplace(0, 1/np.sqrt(2), s)},
    {"name": "Пуассона",   "gen": lambda s: np.random.poisson(5, s)},
    {"name": "Равномерное", "gen": lambda s: np.random.uniform(-np.sqrt(3), np.sqrt(3), s)}
]
sample_sizes = [10, 100, 1000]
repetitions = 1000
stat_names = ["mean", "median", "zR", "zQ", "zTr"]
stat_names_rus = ["Среднее", "Медиана", "z_R", "z_Q", "Усечённое среднее"]

# Подготовка CSV-файла
csv_filename = "results/results.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')
    # Заголовок
    writer.writerow(["Распределение", "Объём выборки", "Статистика", "E(z)", "sqrt(D(z))"])

# Основной цикл
for dist in distributions:
    dname = dist["name"]
    print(f"Обрабатывается {dname} распределение...")
    for n in sample_sizes:
        # Списки для хранения 1000 значений каждой статистики
        stat_vals = [[] for _ in range(5)]
        for _ in range(repetitions):
            sample = dist["gen"](n)
            stats = compute_statistics(sample)
            for i, val in enumerate(stats):
                stat_vals[i].append(val)
        # Вычисляем среднее и дисперсию для каждой статистики
        for i, (sname, sname_rus) in enumerate(zip(stat_names, stat_names_rus)):
            arr = np.array(stat_vals[i])
            mean_val = np.mean(arr)
            var_val = np.var(arr, ddof=0)   # генеральная дисперсия
            std_val = np.sqrt(var_val)
            # Форматирование с одним знаком после запятой и замена "-0.0" на "0.0"
            mean_str = f"{mean_val:.1f}".replace("-0.0", "0.0")
            std_str = f"{std_val:.1f}".replace("-0.0", "0.0")
            # Запись в CSV
            with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow([dname, n, sname_rus, mean_str, std_str])

print(f"\nРезультаты сохранены в файл {csv_filename}")