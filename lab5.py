import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, multivariate_normal
import os

np.random.seed(42)
os.makedirs("images", exist_ok=True)

# ---------- Генераторы выборок ----------
def bivariate_normal(n, rho):
    """Двумерное нормальное N(0,0,1,1,rho)"""
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return np.random.multivariate_normal(mean, cov, n)

def mixture(n):
    """Смесь: 0.9 * N(0,0,1,1,0.9) + 0.1 * N(0,0,10,10,-0.9)"""
    comp1 = bivariate_normal(n, 0.9)
    comp2 = np.random.multivariate_normal([0,0], [[10, -9], [-9, 10]], n)  # дисперсии 10, ковариация -9 → ρ = -0.9
    # Выбор компоненты для каждого наблюдения
    mask = np.random.choice([0,1], size=n, p=[0.9,0.1])
    sample = np.where(mask[:,None], comp2, comp1)
    return sample

# ---------- Коэффициенты корреляции ----------
def pearson_corr(x, y):
    return np.corrcoef(x, y)[0,1]

def spearman_corr(x, y):
    return spearmanr(x, y)[0]

def quadrant_corr(x, y):
    """Квадрантный коэффициент (по медианам)"""
    med_x = np.median(x)
    med_y = np.median(y)
    return np.mean(np.sign((x - med_x) * (y - med_y)))

# ---------- Параметры эксперимента ----------
sample_sizes = [20, 60, 100]
n_reps = 1000

# Определяем сценарии:
scenarios = [
    {"name": "Normal ρ=0.0",   "generator": lambda n: bivariate_normal(n, 0.0),   "rho_true": 0.0},
    {"name": "Normal ρ=0.5",   "generator": lambda n: bivariate_normal(n, 0.5),   "rho_true": 0.5},
    {"name": "Normal ρ=0.9",   "generator": lambda n: bivariate_normal(n, 0.9),   "rho_true": 0.9},
    {"name": "Mixture",        "generator": mixture,                               "rho_true": None}
]

# Результаты: список словарей
results = []

for scenario in scenarios:
    name = scenario["name"]
    gen = scenario["generator"]
    rho_true = scenario["rho_true"]

    for n in sample_sizes:
        print(f"Processing {name}, n={n}")

        # Массивы для коэффициентов
        pearson_vals = []
        spearman_vals = []
        quadrant_vals = []

        for _ in range(n_reps):
            sample = gen(n)
            x = sample[:,0]; y = sample[:,1]
            pearson_vals.append(pearson_corr(x,y))
            spearman_vals.append(spearman_corr(x,y))
            quadrant_vals.append(quadrant_corr(x,y))

        # Средние и дисперсии (генеральная дисперсия)
        mean_pearson = np.mean(pearson_vals)
        var_pearson = np.var(pearson_vals, ddof=0)
        mean_spearman = np.mean(spearman_vals)
        var_spearman = np.var(spearman_vals, ddof=0)
        mean_quadrant = np.mean(quadrant_vals)
        var_quadrant = np.var(quadrant_vals, ddof=0)

        results.append({
            "Scenario": name,
            "n": n,
            "rho_true": rho_true,
            "mean_pearson": mean_pearson,
            "var_pearson": var_pearson,
            "mean_spearman": mean_spearman,
            "var_spearman": var_spearman,
            "mean_quadrant": mean_quadrant,
            "var_quadrant": var_quadrant
        })

        # Для визуализации используем последнюю выборку
        sample = gen(n)
        x = sample[:,0]; y = sample[:,1]

        # Диаграмма рассеяния и эллипс равновероятности
        plt.figure(figsize=(5,4))
        plt.scatter(x, y, alpha=0.6, s=10)

        # Эллипс по выборочным параметрам (уровень 0.5)
        mean_vec = np.mean(sample, axis=0)
        cov_mat = np.cov(sample.T)
        # Вычисляем собственные числа и векторы для угла поворота
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        # Радиус для уровня вероятности p = 0.5: -2 ln(1-p) = -2 ln(0.5) ≈ 1.386
        radius = np.sqrt(-2 * np.log(0.5))  # ~1.177
        width = radius * np.sqrt(eigvals[0])   # длина большой полуоси
        height = radius * np.sqrt(eigvals[1])  # длина малой полуоси
        angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
        ellipse = plt.matplotlib.patches.Ellipse(
            xy=mean_vec, width=2*width, height=2*height, angle=angle,
            edgecolor='r', facecolor='none', linewidth=2, linestyle='--'
        )
        plt.gca().add_patch(ellipse)

        plt.title(f"{name}, n={n}\nПирсон={mean_pearson:.2f} ± {np.sqrt(var_pearson):.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"images/scatter_{name.replace(' ', '_')}_n{n}.png", dpi=150)
        plt.close()

# Сохранение результатов в CSV
import csv
csv_filename = "results/corr_results.csv"
os.makedirs("results", exist_ok=True)
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["Сценарий", "n", "ρ_true", "E[Пирсон]", "D[Пирсон]", "E[Спирмен]", "D[Спирмен]", "E[Квадрантный]", "D[Квадрантный]"])
    for r in results:
        writer.writerow([
            r["Scenario"], r["n"], r["rho_true"] if r["rho_true"] is not None else "",
            f"{r['mean_pearson']:.4f}", f"{r['var_pearson']:.4f}",
            f"{r['mean_spearman']:.4f}", f"{r['var_spearman']:.4f}",
            f"{r['mean_quadrant']:.4f}", f"{r['var_quadrant']:.4f}"
        ])

print(f"Результаты сохранены в {csv_filename}")