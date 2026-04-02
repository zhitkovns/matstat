import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, cauchy, laplace, poisson, uniform

np.random.seed(42)
os.makedirs("images", exist_ok=True)

# ---------- Оценка ширины окна ----------
def silverman_bandwidth(sample):
    n = len(sample)
    return 1.06 * np.std(sample) * n ** (-1 / 5)

def robust_silverman_bandwidth(sample):
    n = len(sample)
    iqr = np.quantile(sample, 0.75) - np.quantile(sample, 0.25)
    return 0.9 * min(np.std(sample), iqr / 1.34) * n ** (-1 / 5)

# ---------- Ядерная оценка (гауссово ядро) ----------
def gaussian_kde(x, sample, h):
    kde = np.zeros_like(x)
    for xi in sample:
        kde += np.exp(-0.5 * ((x - xi) / h) ** 2)
    return kde / (len(sample) * h * np.sqrt(2 * np.pi))

# ---------- Параметры распределений ----------
distributions = [
    {
        "name": "Нормальное",
        "rvs": lambda n: np.random.normal(0, 1, n),
        "cdf": lambda x: norm.cdf(x, 0, 1),
        "pdf": lambda x: norm.pdf(x, 0, 1),
        "xlim": (-4, 4),
        "bw_method": silverman_bandwidth,
        "color": "C0"
    },
    {
        "name": "Коши",
        "rvs": lambda n: np.random.standard_cauchy(n),
        "cdf": lambda x: cauchy.cdf(x, 0, 1),
        "pdf": lambda x: cauchy.pdf(x, 0, 1),
        "xlim": (-100, 100),
        "bw_method": robust_silverman_bandwidth,
        "color": "C1"
    },
    {
        "name": "Лапласа",
        "rvs": lambda n: np.random.laplace(0, 1/np.sqrt(2), n),
        "cdf": lambda x: laplace.cdf(x, 0, 1/np.sqrt(2)),
        "pdf": lambda x: laplace.pdf(x, 0, 1/np.sqrt(2)),
        "xlim": (-4, 4),
        "bw_method": silverman_bandwidth,
        "color": "C2"
    },
    {
        "name": "Пуассона",
        "rvs": lambda n: np.random.poisson(5, n),
        "cdf": lambda x: poisson.cdf(np.floor(x), 5),
        "pmf": lambda k: poisson.pmf(k, 5),
        "xlim": (0, 10),
        "bw_method": silverman_bandwidth,          # добавлен метод для KDE
        "discrete": True,                          # флаг остаётся для корректной отрисовки теоретической "плотности"
        "color": "C3"
    },
    {
        "name": "Равномерное",
        "rvs": lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n),
        "cdf": lambda x: uniform.cdf(x, -np.sqrt(3), 2*np.sqrt(3)),
        "pdf": lambda x: uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)),
        "xlim": (-4, 4),
        "bw_method": silverman_bandwidth,
        "color": "C4"
    }
]

sample_sizes = [20, 60, 100]

for dist in distributions:
    name = dist["name"]
    xlim = dist["xlim"]
    is_discrete = dist.get("discrete", False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Распределение {name}', fontsize=16)

    for idx, n in enumerate(sample_sizes):
        sample = dist["rvs"](n)

        # ----- Подготовка параметров гистограммы -----
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
            # для Пуассона: бины шириной 1, центрированные по целым числам
            bins = np.arange(np.floor(xlim[0]) - 0.5, np.ceil(xlim[1]) + 1.5, 1)
            hist_kwargs['bins'] = bins
            # диапазон не задаём, он определяется бинами
        else:
            hist_kwargs['bins'] = 'fd'
            hist_kwargs['range'] = xlim

        # ----- Верхний ряд: плотность -----
        ax_dens = axes[0, idx]
        # Гистограмма
        ax_dens.hist(sample, **hist_kwargs)

        # Ядерная оценка (теперь для всех распределений, включая Пуассона)
        bw = dist["bw_method"](sample)          # для Пуассона теперь есть метод
        x_dens = np.linspace(xlim[0], xlim[1], 500)
        y_kde = gaussian_kde(x_dens, sample, bw)
        ax_dens.plot(x_dens, y_kde, 'b-', lw=2, label='Ядерная оценка')

        # Теоретическая плотность/вероятность
        if is_discrete:
            k_vals = np.arange(np.floor(xlim[0]), np.ceil(xlim[1])+1)
            pmf_vals = dist["pmf"](k_vals)
            ax_dens.plot(k_vals, pmf_vals, 'ro-', linewidth=1, markersize=4, label='Теоретическая вероятность')
        else:
            y_pdf = dist["pdf"](x_dens)
            ax_dens.plot(x_dens, y_pdf, 'r-', lw=2, label='Теоретическая плотность')

        ax_dens.set_xlim(xlim)
        ax_dens.set_xlabel('x')
        ax_dens.set_ylabel('Плотность')
        ax_dens.set_title(f'n = {n}')
        ax_dens.legend(loc='upper right', fontsize=8)
        ax_dens.grid(alpha=0.3)

        # ----- Нижний ряд: функция распределения -----
        ax_cdf = axes[1, idx]

        # Эмпирическая ФР
        sorted_sample = np.sort(sample)
        y_ecdf = np.arange(1, n+1) / n
        ax_cdf.step(sorted_sample, y_ecdf, where='post', color='blue', lw=2, label='ЭФР')

        # Теоретическая ФР
        x_cdf = np.linspace(xlim[0], xlim[1], 500)
        if is_discrete:
            x_vals = np.arange(np.floor(xlim[0]), np.ceil(xlim[1])+1)
            y_cdf = dist["cdf"](x_vals)
            ax_cdf.step(x_vals, y_cdf, where='post', color='red', lw=2, label='Теоретическая ФР')
        else:
            y_cdf = dist["cdf"](x_cdf)
            ax_cdf.plot(x_cdf, y_cdf, 'r-', lw=2, label='Теоретическая ФР')

        # ЭФР из гистограммы (теперь для всех распределений)
        # Используем те же параметры, что и при построении гистограммы
        if name == "Коши":
            counts, bin_edges = np.histogram(sample, bins=hist_kwargs['bins'], range=hist_kwargs['range'])
        elif is_discrete:
            # Для Пуассона: bins уже заданы явно
            counts, bin_edges = np.histogram(sample, bins=hist_kwargs['bins'])
        else:
            counts, bin_edges = np.histogram(sample, bins='fd', range=xlim)
        cdf_hist = np.cumsum(counts) / n
        ax_cdf.step(bin_edges[1:], cdf_hist, where='post', color='orange', linestyle='--', lw=1.5, label='ЭФР из гистограммы')

        ax_cdf.set_xlim(xlim)
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.set_xlabel('x')
        ax_cdf.set_ylabel('F(x)')
        ax_cdf.legend(loc='lower right', fontsize=8)
        ax_cdf.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'images/{name}.png', dpi=150)
    plt.close()

# ---------- Дополнительные графики (без изменений) ----------
sample_uniform = np.random.uniform(-np.sqrt(3), np.sqrt(3), 100)
x = np.linspace(-4, 4, 500)
h_silver_uniform = silverman_bandwidth(sample_uniform)

plt.figure(figsize=(8, 5))
plt.plot(x, uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)), 'r-', lw=2, label='Теоретическая плотность')
for scale, style in zip([0.5, 1.0, 2.0], ['--', '-', ':']):
    h = scale * h_silver_uniform
    y_kde = gaussian_kde(x, sample_uniform, h)
    plt.plot(x, y_kde, color='b', linestyle=style, lw=1.5, label=f'KDE, h = {scale:.1f}·h_silver')
plt.xlim(-4, 4)
plt.xlabel('x')
plt.ylabel('Плотность')
plt.title('Влияние ширины окна на ядерную оценку (равномерное, n=100)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/kde_bandwidth_effect_uniform.png', dpi=150)
plt.close()

sample_laplace = np.random.laplace(0, 1/np.sqrt(2), 100)
x = np.linspace(-4, 4, 500)
h_silver_laplace = silverman_bandwidth(sample_laplace)

plt.figure(figsize=(8, 5))
plt.plot(x, laplace.pdf(x, 0, 1/np.sqrt(2)), 'r-', lw=2, label='Теоретическая плотность')
for scale, style in zip([0.5, 1.0, 2.0], ['--', '-', ':']):
    h = scale * h_silver_laplace
    y_kde = gaussian_kde(x, sample_laplace, h)
    plt.plot(x, y_kde, color='b', linestyle=style, lw=1.5, label=f'KDE, h = {scale:.1f}·h_silver')
plt.xlim(-4, 4)
plt.xlabel('x')
plt.ylabel('Плотность')
plt.title('Влияние ширины окна на ядерную оценку (Лапласа, n=100)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/kde_bandwidth_effect_laplace.png', dpi=150)
plt.close()

print("Графики сохранены в папке images/")