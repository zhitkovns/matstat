import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2, f
import os

np.random.seed(42)
os.makedirs("images", exist_ok=True)

# ---------- Генерация двух независимых выборок ----------
n1 = 20
n2 = 100
sample1 = np.random.normal(0, 1, n1)
sample2 = np.random.normal(0, 1, n2)

# Точечные оценки
mean1 = np.mean(sample1)
mean2 = np.mean(sample2)
var1 = np.var(sample1, ddof=1)   # несмещённая дисперсия
var2 = np.var(sample2, ddof=1)
std1 = np.sqrt(var1)
std2 = np.sqrt(var2)

print("=== Точечные оценки ===")
print(f"Выборка 1 (n={n1}): среднее = {mean1:.4f}, дисперсия = {var1:.4f}")
print(f"Выборка 2 (n={n2}): среднее = {mean2:.4f}, дисперсия = {var2:.4f}")

# ---------- Доверительные интервалы для среднего (t-распределение) ----------
alpha = 0.05
t_crit = t.ppf(1 - alpha/2, df=n1-1)
moe1 = t_crit * std1 / np.sqrt(n1)
ci_mean1 = (mean1 - moe1, mean1 + moe1)

t_crit2 = t.ppf(1 - alpha/2, df=n2-1)
moe2 = t_crit2 * std2 / np.sqrt(n2)
ci_mean2 = (mean2 - moe2, mean2 + moe2)

print("\n=== Доверительные интервалы для среднего (уровень 0.95) ===")
print(f"Выборка 1: [{ci_mean1[0]:.4f}, {ci_mean1[1]:.4f}]")
print(f"Выборка 2: [{ci_mean2[0]:.4f}, {ci_mean2[1]:.4f}]")

# ---------- Доверительные интервалы для дисперсии (χ²-распределение) ----------
chi2_low = chi2.ppf(alpha/2, df=n1-1)
chi2_high = chi2.ppf(1 - alpha/2, df=n1-1)
ci_var1 = ((n1-1)*var1 / chi2_high, (n1-1)*var1 / chi2_low)

chi2_low2 = chi2.ppf(alpha/2, df=n2-1)
chi2_high2 = chi2.ppf(1 - alpha/2, df=n2-1)
ci_var2 = ((n2-1)*var2 / chi2_high2, (n2-1)*var2 / chi2_low2)

print("\n=== Доверительные интервалы для дисперсии (уровень 0.95) ===")
print(f"Выборка 1: [{ci_var1[0]:.4f}, {ci_var1[1]:.4f}]")
print(f"Выборка 2: [{ci_var2[0]:.4f}, {ci_var2[1]:.4f}]")

# ---------- F-тест для равенства дисперсий ----------
# Помещаем большую дисперсию в числитель
if var1 >= var2:
    F_stat = var1 / var2
    df_num = n1 - 1
    df_den = n2 - 1
    label = "s1^2 / s2^2"
else:
    F_stat = var2 / var1
    df_num = n2 - 1
    df_den = n1 - 1
    label = "s2^2 / s1^2"

# Критическое значение для двусторонней альтернативы (α=0.05)
f_crit = f.ppf(1 - alpha/2, df_num, df_den)

print("\n=== F-тест (равенство дисперсий) ===")
print(f"Статистика F = {F_stat:.4f} ({label})")
print(f"Степени свободы: {df_num}, {df_den}")
print(f"Критическое значение (α=0.05, двустороннее): {f_crit:.4f}")

if F_stat > f_crit:
    print("Гипотеза о равенстве дисперсий ОТВЕРГАЕТСЯ на уровне 0.05")
else:
    print("Гипотеза о равенстве дисперсий НЕ отвергается на уровне 0.05")

# Дополнительно: односторонний тест (если нужно)
f_crit_one = f.ppf(1 - alpha, df_num, df_den)
print(f"Одностороннее критическое значение (α=0.05): {f_crit_one:.4f}")
if F_stat > f_crit_one:
    print("В односторонней альтернативе (большая дисперсия больше) гипотеза отвергается")
else:
    print("В односторонней альтернативе гипотеза не отвергается")

# ---------- Визуализация выборок ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(sample1, bins='auto', density=True, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title(f'Выборка 1 (n={n1}) из N(0,1)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Плотность')
axes[0].grid(alpha=0.3)

axes[1].hist(sample2, bins='auto', density=True, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title(f'Выборка 2 (n={n2}) из N(0,1)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Плотность')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/samples_hist.png', dpi=150)
plt.close()

print("\nГистограммы сохранены в images/samples_hist.png")