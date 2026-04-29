import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

np.random.seed(42)
os.makedirs("images", exist_ok=True)

def bivariate_normal(n, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return np.random.multivariate_normal(mean, cov, n)

def mixture(n):
    comp1 = bivariate_normal(n, 0.9)
    comp2 = np.random.multivariate_normal([0,0], [[10, -9], [-9, 10]], n)
    mask = np.random.choice([0,1], size=n, p=[0.9,0.1])
    return np.where(mask[:,None], comp2, comp1)

def pearson_corr(x, y):
    return np.corrcoef(x, y)[0,1]

def spearman_corr(x, y):
    return spearmanr(x, y)[0]

def quadrant_corr(x, y):
    med_x, med_y = np.median(x), np.median(y)
    return np.mean(np.sign((x - med_x) * (y - med_y)))

sample_sizes = [20, 60, 100]
n_reps = 1000

scenarios = [
    {"name": "Normal_rho0.0",   "gen": lambda n: bivariate_normal(n, 0.0),   "true_rho": 0.0, "xlim": (-4,4), "ylim": (-4,4)},
    {"name": "Normal_rho0.5",   "gen": lambda n: bivariate_normal(n, 0.5),   "true_rho": 0.5, "xlim": (-4,4), "ylim": (-4,4)},
    {"name": "Normal_rho0.9",   "gen": lambda n: bivariate_normal(n, 0.9),   "true_rho": 0.9, "xlim": (-4,4), "ylim": (-4,4)},
    {"name": "Mixture",         "gen": mixture,                               "true_rho": None, "xlim": (-5,5), "ylim": (-5,5)}
]

results = []

for scen in scenarios:
    name, gen, true_rho, xlim, ylim = scen["name"], scen["gen"], scen["true_rho"], scen["xlim"], scen["ylim"]
    for n in sample_sizes:
        print(f"Processing {name}, n={n}")
        pv, sv, qv = [], [], []
        for _ in range(n_reps):
            samp = gen(n)
            x, y = samp[:,0], samp[:,1]
            pv.append(pearson_corr(x,y))
            sv.append(spearman_corr(x,y))
            qv.append(quadrant_corr(x,y))
        results.append({
            "Scenario": name, "n": n, "rho_true": true_rho,
            "mean_pearson": np.mean(pv), "var_pearson": np.var(pv, ddof=0),
            "mean_spearman": np.mean(sv), "var_spearman": np.var(sv, ddof=0),
            "mean_quadrant": np.mean(qv), "var_quadrant": np.var(qv, ddof=0)
        })

        # ---- Визуализация ----
        samp = gen(n)
        x, y = samp[:,0], samp[:,1]
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(x, y, alpha=0.6, s=10)

        # Выбираем точки, по которым строим эллипс
        if "Mixture" in name:
            # Отбираем точки в пределах [-3.5, 3.5] (основная масса)
            mask = (np.abs(x) < 3.5) & (np.abs(y) < 3.5)
            samp_ellipse = samp[mask]
        else:
            samp_ellipse = samp

        # Если осталось хотя бы 3 точки, строим эллипс
        if len(samp_ellipse) > 2:
            mean_vec = np.mean(samp_ellipse, axis=0)
            cov_mat = np.cov(samp_ellipse.T)
            eigvals, eigvecs = np.linalg.eigh(cov_mat)
            radius = np.sqrt(-2 * np.log(0.5))          # 1.177 для p=0.5
            width  = radius * np.sqrt(eigvals[0])
            height = radius * np.sqrt(eigvals[1])
            angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
            ellipse = plt.matplotlib.patches.Ellipse(
                xy=mean_vec, width=2*width, height=2*height, angle=angle,
                edgecolor='r', facecolor='none', linewidth=2, linestyle='--'
            )
            ax.add_patch(ellipse)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f"{name.replace('_',' ')}, n={n}\nПирсон={np.mean(pv):.2f} ± {np.std(pv):.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"images/scatter_{name}_n{n}.png", dpi=150)
        plt.close()

# ---- CSV ----
import csv
os.makedirs("results", exist_ok=True)
with open("results/corr_results.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["Сценарий","n","ρ_true","E[Пирсон]","D[Пирсон]","E[Спирмен]","D[Спирмен]","E[Квадрантный]","D[Квадрантный]"])
    for r in results:
        writer.writerow([
            r["Scenario"], r["n"], r["rho_true"] if r["rho_true"] is not None else "",
            f"{r['mean_pearson']:.4f}", f"{r['var_pearson']:.4f}",
            f"{r['mean_spearman']:.4f}", f"{r['var_spearman']:.4f}",
            f"{r['mean_quadrant']:.4f}", f"{r['var_quadrant']:.4f}"
        ])
print("Готово.")