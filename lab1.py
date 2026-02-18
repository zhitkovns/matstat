import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, uniform, poisson

np.random.seed(42)

n_values = [10, 100, 1000]

distributions = {
    'normal': {
        'generator': np.random.normal,
        'gen_args': (0, 1),
        'pdf_func': norm.pdf,
        'pdf_args': (0, 1),
        'x_range': (-4, 4),
        'discrete': False,
        'title': r'Нормальное $N(0,1)$'
    },
    'cauchy': {
        'generator': np.random.standard_cauchy,
        'gen_args': (),
        'pdf_func': cauchy.pdf,
        'pdf_args': (0, 1),
        'x_range': (-20, 20),
        'discrete': False,
        'title': r'Коши $C(0,1)$'
    },
    'laplace': {
        'generator': np.random.laplace,
        'gen_args': (0, 1/np.sqrt(2)),
        'pdf_func': laplace.pdf,
        'pdf_args': (0, 1/np.sqrt(2)),
        'x_range': (-6, 6),
        'discrete': False,
        'title': r'Лапласа $L(0,1/\sqrt{2})$'
    },
    'poisson': {
        'generator': np.random.poisson,
        'gen_args': (10,),
        'pmf_func': poisson.pmf,
        'pmf_args': (10,),
        'x_range': (0, 20),
        'discrete': True,
        'title': r'Пуассона $P(10)$'
    },
    'uniform': {
        'generator': np.random.uniform,
        'gen_args': (-np.sqrt(3), np.sqrt(3)),
        'pdf_func': uniform.pdf,
        'pdf_args': (-np.sqrt(3), 2*np.sqrt(3)),
        'x_range': (-2, 2),
        'discrete': False,
        'title': r'Равномерное $U(-\sqrt{3},\sqrt{3})$'
    }
}

for name, dist in distributions.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Распределение: {dist["title"]}', fontsize=14)

    for i, n in enumerate(n_values):
        ax = axes[i]
        data = dist['generator'](*dist['gen_args'], n)

        if dist['discrete']:
            # Пуассон
            if n <= 100:
                # Для малых выборок используем автоматический подбор бинов
                bins = 'auto'
                # Ограничиваем диапазон, чтобы избежать лишних хвостов
                ax.hist(data, bins=bins, range=(0, 20), density=True, alpha=0.7,
                        edgecolor='black', label='Гистограмма')
            else:
                # Для n=1000 используем точные бины шириной 1
                bins = np.arange(-0.5, 20.5, 1)
                ax.hist(data, bins=bins, density=True, alpha=0.7, edgecolor='black',
                        label='Гистограмма')
            # Теоретические плотности
            x_vals = np.arange(0, 21)
            pmf_vals = dist['pmf_func'](x_vals, *dist['pmf_args'])
            ax.plot(x_vals, pmf_vals, 'ro-', markersize=4, label='Теоретическая вероятность')
            ax.set_xlim(0, 20)
            ax.set_xlabel('k')
        else:
            # Непрерывные распределения
            if name == 'cauchy':
                if n == 10:
                    hist_range = (-10, 10)
                    bins = 20
                elif n == 100:
                    hist_range = (-15, 15)
                    bins = 30
                else:
                    hist_range = (-20, 20)
                    bins = 100
                ax.hist(data, bins=bins, range=hist_range, density=True, alpha=0.7,
                        edgecolor='black', label='Гистограмма')
                ax.set_xlim(hist_range)
                x_vals = np.linspace(hist_range[0], hist_range[1], 200)
                pdf_vals = dist['pdf_func'](x_vals, *dist['pdf_args'])
                ax.plot(x_vals, pdf_vals, 'r-', label='Теоретическая плотность')
            else:
                if n <= 100:
                    bins = 'sqrt'
                else:
                    bins = 'auto'
                ax.hist(data, bins=bins, density=True, alpha=0.7, edgecolor='black',
                        label='Гистограмма')
                ax.set_xlim(dist['x_range'])
                x_vals = np.linspace(dist['x_range'][0], dist['x_range'][1], 200)
                pdf_vals = dist['pdf_func'](x_vals, *dist['pdf_args'])
                ax.plot(x_vals, pdf_vals, 'r-', label='Теоретическая плотность')
            ax.set_xlabel('x')

        ax.set_ylabel('Плотность')
        ax.set_title(f'n = {n}')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'{name}.png', dpi=150)
    plt.show()