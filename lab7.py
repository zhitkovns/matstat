import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, uniform, laplace
import os

np.random.seed(42)
os.makedirs("images", exist_ok=True)

# ---------- 1. Основная выборка из N(0,1) ----------
n_main = 100
sample = np.random.normal(0, 1, n_main)

# Оценка параметров ММП (совпадает с МНК для среднего)
mu_mle = np.mean(sample)
sigma_mle = np.std(sample, ddof=0)   # смещённая оценка (деление на n)

print("=== Основная выборка ===")
print(f"Выборочное среднее = {mu_mle:.4f}")
print(f"Выборочное СКО (ММП) = {sigma_mle:.4f}")

# ---------- Построение интервалов (равновероятностные) ----------
k = 8  # число интервалов
p_edges = np.linspace(0, 1, k+1)
quantiles = norm.ppf(p_edges, loc=mu_mle, scale=sigma_mle)
expected = n_main / k   # 12.5

# Наблюдаемые частоты
observed = np.histogram(sample, bins=quantiles)[0]

# Статистика хи-квадрат
chi2_stat = np.sum((observed - expected)**2 / expected)

df = k - 1 - 2   # степени свободы: k-1 минус 2 оценённых параметра
critical = chi2.ppf(0.95, df)

print(f"\nЧисло интервалов k = {k}")
print(f"Степени свободы df = {df}")
print(f"χ² = {chi2_stat:.4f}")
print(f"Критическое значение (α=0.05) = {critical:.4f}")

if chi2_stat < critical:
    print("Гипотеза о нормальном распределении НЕ отвергается")
else:
    print("Гипотеза о нормальном распределении ОТВЕРГАЕТСЯ")

# ---------- Таблица для отчёта (интервалы) ----------
print("\n=== Таблица интервалов ===")
print("№\tГраницы интервала\t\tni\tnpi\t(ni-npi)^2/(npi)")
for i in range(k):
    left = quantiles[i]
    right = quantiles[i+1]
    if i == 0:
        interval_str = f"(-∞, {right:.2f}]"
    elif i == k-1:
        interval_str = f"[{left:.2f}, +∞)"
    else:
        interval_str = f"[{left:.2f}, {right:.2f}]"
    print(f"{i+1}\t{interval_str}\t\t{observed[i]}\t{expected:.1f}\t{(observed[i]-expected)**2/expected:.4f}")

# ---------- Визуализация ----------
plt.figure(figsize=(6,4))
plt.hist(sample, bins='sqrt', density=True, alpha=0.5, color='gray', label='Выборка')
x_vals = np.linspace(-3, 3, 200)
plt.plot(x_vals, norm.pdf(x_vals, mu_mle, sigma_mle), 'r-', label='Нормальная плотность (ММП)')
plt.title('Выборка из N(0,1) и оценённая нормальная плотность')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/main_sample.png', dpi=150)
plt.close()

# ---------- 2. Исследование мощности критерия ----------
n_alt = 20
n_reps = 1000
k_alt = 4   # чтобы np_i = 5 ≥5
expected_alt = n_alt / k_alt
df_alt = k_alt - 1 - 2   # 1

def test_normality(sample):
    mu = np.mean(sample)
    sigma = np.std(sample, ddof=0)
    p_edges = np.linspace(0, 1, k_alt+1)
    quantiles = norm.ppf(p_edges, loc=mu, scale=sigma)
    obs = np.histogram(sample, bins=quantiles)[0]
    chi2_val = np.sum((obs - expected_alt)**2 / expected_alt)
    crit = chi2.ppf(0.95, df_alt)
    return chi2_val >= crit

reject_uniform = 0
for _ in range(n_reps):
    samp = np.random.uniform(-np.sqrt(3), np.sqrt(3), n_alt)
    if test_normality(samp):
        reject_uniform += 1
power_uniform = reject_uniform / n_reps

reject_laplace = 0
for _ in range(n_reps):
    samp = np.random.laplace(0, 1/np.sqrt(2), n_alt)
    if test_normality(samp):
        reject_laplace += 1
power_laplace = reject_laplace / n_reps

print("\n=== Исследование мощности критерия ===")
print(f"Объём выборки для альтернатив = {n_alt}, k = {k_alt}, ожидаемая частота = {expected_alt}")
print(f"Доля отвергнутых H0 (равномерное): {power_uniform*100:.2f}%")
print(f"Доля отвергнутых H0 (Лапласа): {power_laplace*100:.2f}%")

# Визуализация примеров
fig, axes = plt.subplots(1, 2, figsize=(10,4))
samp_unif = np.random.uniform(-np.sqrt(3), np.sqrt(3), n_alt)
samp_lapl = np.random.laplace(0, 1/np.sqrt(2), n_alt)

axes[0].hist(samp_unif, bins='sqrt', density=True, alpha=0.5, color='blue')
x = np.linspace(-2, 2, 200)
axes[0].plot(x, uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)), 'b-', label='Равномерная плотность')
axes[0].set_title(f'Равномерное (n={n_alt})')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(samp_lapl, bins='sqrt', density=True, alpha=0.5, color='green')
x = np.linspace(-3, 3, 200)
axes[1].plot(x, laplace.pdf(x, 0, 1/np.sqrt(2)), 'g-', label='Лапласа плотность')
axes[1].set_title(f'Лапласа (n={n_alt})')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/alternative_samples.png', dpi=150)
plt.close()

print("\nГрафики сохранены в images/")