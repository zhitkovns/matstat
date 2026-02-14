import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, uniform, poisson

# Параметры
n_values = [10, 100, 1000]

# Описание распределений
distributions = {
    'normal': {
        'generator': np.random.normal,
        'gen_args': (0, 1),
        'pdf_func': norm.pdf,
        'pdf_args': (0, 1),
        'x_range': (-4, 4),
        'discrete': False,
        'title': 'Нормальное N(0,1)'
    },
    'cauchy': {
        'generator': np.random.standard_cauchy,
        'gen_args': (),
        'pdf_func': cauchy.pdf,
        'pdf_args': (0, 1),
        'x_range': (-35, 35),
        'discrete': False,
        'title': 'Коши C(0,1)'
    },
    'laplace': {
        'generator': np.random.laplace,
        'gen_args': (0, 1/np.sqrt(2)),
        'pdf_func': laplace.pdf,
        'pdf_args': (0, 1/np.sqrt(2)),
        'x_range': (-6, 6),
        'discrete': False,
        'title': 'Лапласа L(0, 1/√2)'
    },
    'poisson': {
        'generator': np.random.poisson,
        'gen_args': (10,),
        'pmf_func': poisson.pmf,
        'pmf_args': (10,),
        'x_range': (0, 20),
        'discrete': True,
        'title': 'Пуассона P(10)'
    },
    'uniform': {
        'generator': np.random.uniform,
        'gen_args': (-np.sqrt(3), np.sqrt(3)),
        'pdf_func': uniform.pdf,
        'pdf_args': (-np.sqrt(3), 2*np.sqrt(3)),
        'x_range': (-2, 2),
        'discrete': False,
        'title': 'Равномерное U(−√3, √3)'
    }
}

for name, dist in distributions.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Распределение: {dist["title"]}', fontsize=14)

    for i, n in enumerate(n_values):
        ax = axes[i]

        # Генерация выборки
        data = dist['generator'](*dist['gen_args'], n)

        if dist['discrete']:
            # Пуассон – дискретное распределение
            bins = np.arange(-0.5, 20.5, 1)  # фиксированные бины для целых значений
            ax.hist(data, bins=bins, density=True, alpha=0.7, edgecolor='black', label='Гистограмма') # density - это нормировка, чтобы площадь под графиком была равна 1
            # Теоретические вероятности
            x_vals = np.arange(0, 21)
            pmf_vals = dist['pmf_func'](x_vals, *dist['pmf_args'])
            ax.plot(x_vals, pmf_vals, 'ro-', markersize=4, label='Теоретическая вероятность')
            ax.set_xlabel('k')
        else:
            # Непрерывные распределения
            ax.hist(data, bins='auto', density=True, alpha=0.7, edgecolor='black', label='Гистограмма') # density - это нормировка, чтобы площадь под графиком была равна 1
            x_vals = np.linspace(dist['x_range'][0], dist['x_range'][1], 200)
            pdf_vals = dist['pdf_func'](x_vals, *dist['pdf_args'])
            ax.plot(x_vals, pdf_vals, 'r-', label='Теоретическая плотность')
            ax.set_xlabel('x')
        
        ax.set_xlim(dist['x_range'])
        ax.set_ylabel('Плотность')
        ax.set_title(f'n = {n}')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'{name}.png', dpi=150)
    plt.show()