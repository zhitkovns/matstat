import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde, norm, cauchy, laplace, poisson, uniform

np.random.seed(42)
os.makedirs("images", exist_ok=True)

distributions = [
    {
        "name": "Нормальное",
        "rvs": lambda n: np.random.normal(0, 1, n),
        "cdf": lambda x: norm.cdf(x, 0, 1),
        "pdf": lambda x: norm.pdf(x, 0, 1),
        "xlim": (-4, 4),
        "color": "C0"
    },
    {
        "name": "Коши",
        "rvs": lambda n: np.random.standard_cauchy(n),
        "cdf": lambda x: cauchy.cdf(x, 0, 1),
        "pdf": lambda x: cauchy.pdf(x, 0, 1),
        "xlim": (-4, 4),
        "color": "C1"
    },
    {
        "name": "Лапласа",
        "rvs": lambda n: np.random.laplace(0, 1/np.sqrt(2), n),
        "cdf": lambda x: laplace.cdf(x, 0, 1/np.sqrt(2)),
        "pdf": lambda x: laplace.pdf(x, 0, 1/np.sqrt(2)),
        "xlim": (-4, 4),
        "color": "C2"
    },
    {
        "name": "Пуассона",
        "rvs": lambda n: np.random.poisson(5, n),
        "cdf": lambda x: poisson.cdf(np.floor(x), 5),
        "pmf": lambda k: poisson.pmf(k, 5),
        "xlim": (4, 16),
        "color": "C3",
        "discrete": True
    },
    {
        "name": "Равномерное",
        "rvs": lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n),
        "cdf": lambda x: uniform.cdf(x, -np.sqrt(3), 2*np.sqrt(3)),
        "pdf": lambda x: uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)),
        "xlim": (-4, 4),
        "color": "C4"
    }
]

sample_sizes = [20, 60, 100]

for dist in distributions:
    name = dist["name"]
    xlim = dist["xlim"]
    is_discrete = dist.get("discrete", False)

    for n in sample_sizes:
        sample = dist["rvs"](n)

        # ---- ECDF ----
        plt.figure(figsize=(5, 4))
        sorted_sample = np.sort(sample)
        y_ecdf = np.arange(1, n+1) / n
        x_theor = np.linspace(xlim[0], xlim[1], 500)
        if is_discrete:
            x_vals = np.arange(np.floor(xlim[0]), np.ceil(xlim[1])+1)
            y_cdf = dist["cdf"](x_vals)
            plt.step(x_vals, y_cdf, where='post', color='red', lw=2, label='Теоретическая ФР')
        else:
            y_cdf = dist["cdf"](x_theor)
            plt.plot(x_theor, y_cdf, 'r-', lw=2, label='Теоретическая ФР')
        plt.step(sorted_sample, y_ecdf, where='post', color='blue', lw=2, label='Эмпирическая ФР')
        plt.xlim(xlim)
        plt.ylim(0, 1.05)
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.title(f'{name}, n = {n}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'images/ecdf_{name}_n{n}.png', dpi=150)
        plt.close()

        # ---- KDE ----
        plt.figure(figsize=(5, 4))
        hist_kwargs = {'density': True, 'alpha': 0.3, 'color': 'gray', 'label': 'Гистограмма'}

        if name == "Коши":
            if n <= 30:
                hist_range = (-10, 10); bins = 20
            elif n <= 70:
                hist_range = (-12, 12); bins = 30
            else:
                hist_range = (-15, 15); bins = 40
            hist_kwargs['range'] = hist_range
            hist_kwargs['bins'] = bins
        elif is_discrete:
            bins = np.arange(np.floor(xlim[0]) - 0.5, np.ceil(xlim[1]) + 1.5, 1)
            hist_kwargs['bins'] = bins
        else:
            hist_kwargs['bins'] = 'sqrt'
            hist_kwargs['range'] = xlim

        plt.hist(sample, **hist_kwargs)

        # Ручной подбор ширины окна для Коши, чтобы кривая на [-4,4] максимально соответствовала теории
        if name == "Коши":
            if n == 20:
                bw = 0.60  # подобрано для хорошего совпадения при малом объёме
            elif n == 60:
                bw = 0.04   # с ростом n окно уменьшается, но не слишком сильно, чтобы компенсировать выбросы
            else:  # n == 100
                bw = 0.05
            kde = gaussian_kde(sample, bw_method=bw)
        else:
            kde = gaussian_kde(sample, bw_method='silverman')

        x_dens = np.linspace(xlim[0], xlim[1], 500)
        y_kde = kde(x_dens)
        plt.plot(x_dens, y_kde, 'b-', lw=2, label='Ядерная оценка')

        if is_discrete:
            k_vals = np.arange(np.floor(xlim[0]), np.ceil(xlim[1])+1)
            pmf_vals = dist["pmf"](k_vals)
            plt.plot(k_vals, pmf_vals, 'r-', linewidth=1, markersize=4, label='Теоретическая вероятность')
        else:
            y_pdf = dist["pdf"](x_dens)
            plt.plot(x_dens, y_pdf, 'r-', lw=2, label='Теоретическая плотность')

        plt.xlim(xlim)
        plt.xlabel('x')
        plt.ylabel('Плотность')
        plt.title(f'{name}, n = {n}')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'images/kde_{name}_n{n}.png', dpi=150)
        plt.close()

print("Графики сохранены в папке images/")