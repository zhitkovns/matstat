import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Устанавливаем seed для воспроизводимости
np.random.seed(42)

# Создаём папку для результатов
os.makedirs("results", exist_ok=True)
os.makedirs("images", exist_ok=True)

def compute_outlier_fraction(sample):
    """Возвращает долю выбросов по правилу Тьюки"""
    q1 = np.percentile(sample, 25)
    q3 = np.percentile(sample, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = (sample < lower) | (sample > upper)
    return np.mean(outliers)

# Параметры эксперимента
distributions = [
    {"name": "Нормальное", 
     "gen": lambda s: np.random.normal(0, 1, s),
     "color": "C0",
     "xlim": (-4, 4)},
    {"name": "Коши",
     "gen": lambda s: np.random.standard_cauchy(s),
     "color": "C1",
     "xlim": (-15, 15)},
    {"name": "Лапласа",
     "gen": lambda s: np.random.laplace(0, 1/np.sqrt(2), s),
     "color": "C2",
     "xlim": (-4, 4)},
    {"name": "Пуассона",
     "gen": lambda s: np.random.poisson(5, s),
     "color": "C3",
     "xlim": (-2, 12)},
    {"name": "Равномерное",
     "gen": lambda s: np.random.uniform(-np.sqrt(3), np.sqrt(3), s),
     "color": "C4",
     "xlim": (-2, 2)}
]
sample_sizes = [20, 100]
repetitions = 1000

# CSV для результатов
csv_filename = "results/outlier_fractions.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["Распределение", "n", "Средняя доля выбросов"])

# Основной цикл
for dist in distributions:
    dname = dist["name"]
    print(f"Обрабатывается {dname} распределение...")
    for n in sample_sizes:
        fractions = []
        # Генерируем 1000 выборок и собираем доли выбросов
        for _ in range(repetitions):
            sample = dist["gen"](n)
            frac = compute_outlier_fraction(sample)
            fractions.append(frac)
        avg_frac = np.mean(fractions)
        # Запись в CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([dname, n, f"{avg_frac:.4f}"])
        # Построение боксплота для последней выборки (для визуализации)
        plt.figure(figsize=(5, 3))
        plt.boxplot(sample, vert=False, patch_artist=True,
                    boxprops=dict(facecolor=dist["color"]),
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red',
                                    markersize=4, linestyle='none',
                                    markeredgecolor='red'))
        plt.title(f"{dname} (n={n})")
        plt.xlabel("Значения")
        plt.xlim(dist["xlim"])  # фиксируем интервал для данного распределения
        plt.tight_layout()
        plt.savefig(f"images/boxplot_{dname}_n{n}.png", dpi=150)
        plt.close()

print(f"\nРезультаты сохранены в {csv_filename}")
print("Изображения боксплотов сохранены в папке images/")