import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Генерация данных ----------
x = np.arange(-1.8, 2.01, 0.2)  # 20 точек
beta0_true, beta1_true = 2.0, 2.0
eps = np.random.normal(0, 1, len(x))
y = beta0_true + beta1_true * x + eps

# ---------- МНК (через аналитические формулы) ----------
def ols(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_mean = np.mean(x * y)
    x2_mean = np.mean(x * x)
    beta1 = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean**2)
    beta0 = y_mean - beta1 * x_mean
    return beta0, beta1

# ---------- Робастные оценки МНМ по методичке ----------
def robust_mnm(x, y):
    n = len(x)
    # Медианы
    med_x = np.median(x)
    med_y = np.median(y)

    # Знаковый (квадрантный) коэффициент корреляции r_Q
    sign = np.sign
    r_q = np.mean(sign(x - med_x) * sign(y - med_y))

    # Нижний и верхний квартили по дискретному правилу
    # Определяем индексы l и j (порядковые статистики)
    # Сортируем данные отдельно для x и y
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # Вычисляем l и j по формуле из методички
    if n % 4 == 0:          # n/4 целое
        l = n // 4
    else:
        l = n // 4 + 1       # [n/4] + 1
    # l – номер нижнего квартиля (1-индексация)
    # В python индексы смещены на 1, поэтому при обращении используем l-1
    j = n - l + 1           # номер верхнего квартиля

    # Интерквартильные широты (размах) до нормировки
    iqr_x = x_sorted[j-1] - x_sorted[l-1]   # x_{(j)} - x_{(l)}
    iqr_y = y_sorted[j-1] - y_sorted[l-1]

    # Нормировочный коэффициент k_q(n) – для простоты берём 1.34
    # (теоретическое отношение IQR/σ для нормального распределения)
    k_q = 1.34
    qx = iqr_x / k_q
    qy = iqr_y / k_q

    # Робастные оценки коэффициентов
    beta1_rob = r_q * (qy / qx) if qx != 0 else 0.0
    beta0_rob = med_y - beta1_rob * med_x
    return beta0_rob, beta1_rob

# ---------- Без выбросов ----------
beta0_ols, beta1_ols = ols(x, y)
beta0_mnm, beta1_mnm = robust_mnm(x, y)

# Относительные погрешности
def rel_error(true, est):
    return np.abs((true - est) / true) * 100

err_b0_ols = rel_error(beta0_true, beta0_ols)
err_b1_ols = rel_error(beta1_true, beta1_ols)
err_b0_mnm = rel_error(beta0_true, beta0_mnm)
err_b1_mnm = rel_error(beta1_true, beta1_mnm)

print("=== Без выбросов ===")
print(f"МНК: β0 = {beta0_ols:.4f}, β1 = {beta1_ols:.4f}")
print(f"МНМ: β0 = {beta0_mnm:.4f}, β1 = {beta1_mnm:.4f}")

# ---------- С выбросами ----------
y_out = y.copy()
y_out[0] += 10
y_out[-1] -= 10

beta0_ols_out, beta1_ols_out = ols(x, y_out)
beta0_mnm_out, beta1_mnm_out = robust_mnm(x, y_out)

err_b0_ols_out = rel_error(beta0_true, beta0_ols_out)
err_b1_ols_out = rel_error(beta1_true, beta1_ols_out)
err_b0_mnm_out = rel_error(beta0_true, beta0_mnm_out)
err_b1_mnm_out = rel_error(beta1_true, beta1_mnm_out)

print("\n=== С выбросами ===")
print(f"МНК: β0 = {beta0_ols_out:.4f}, β1 = {beta1_ols_out:.4f}")
print(f"МНМ: β0 = {beta0_mnm_out:.4f}, β1 = {beta1_mnm_out:.4f}")

# ---------- Построение графиков с одинаковыми границами ----------
x_line = np.linspace(-2, 2.2, 100)
y_true_line = beta0_true + beta1_true * x_line
y_ols_line = beta0_ols + beta1_ols * x_line
y_mnm_line = beta0_mnm + beta1_mnm * x_line
y_ols_out_line = beta0_ols_out + beta1_ols_out * x_line
y_mnm_out_line = beta0_mnm_out + beta1_mnm_out * x_line

all_y = np.concatenate([y, y_out, y_true_line, y_ols_line, y_mnm_line, y_ols_out_line, y_mnm_out_line])
x_min, x_max = -2.2, 2.2
y_min, y_max = all_y.min(), all_y.max()
y_pad = (y_max - y_min) * 0.1
y_min -= y_pad
y_max += y_pad

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Без выбросов
ax = axes[0]
ax.scatter(x, y, color='blue', label='Данные')
ax.plot(x_line, y_true_line, 'k-', label='Истинная прямая')
ax.plot(x_line, y_ols_line, 'r--', label=f'МНК: β0={beta0_ols:.2f}, β1={beta1_ols:.2f}')
ax.plot(x_line, y_mnm_line, 'g--', label=f'МНМ: β0={beta0_mnm:.2f}, β1={beta1_mnm:.2f}')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Без выбросов')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# С выбросами
ax = axes[1]
ax.scatter(x, y_out, color='blue', label='Данные с выбросами')
ax.plot(x_line, y_true_line, 'k-', label='Истинная прямая')
ax.plot(x_line, y_ols_out_line, 'r--', label=f'МНК: β0={beta0_ols_out:.2f}, β1={beta1_ols_out:.2f}')
ax.plot(x_line, y_mnm_out_line, 'g--', label=f'МНМ: β0={beta0_mnm_out:.2f}, β1={beta1_mnm_out:.2f}')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('С выбросами')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression.png', dpi=150)
plt.show()

# ---------- Вывод таблицы для отчёта ----------
print("\nРезультаты в виде таблицы:")
print("|        | β0   | Δβ0   | δβ0, % | β1   | Δβ1   | δβ1, % |")
print("|--------|------|-------|--------|------|-------|--------|")
print(f"| МНК    | {beta0_ols:.3f} | {beta0_ols - beta0_true:.3f} | {err_b0_ols:.2f} | {beta1_ols:.3f} | {beta1_ols - beta1_true:.3f} | {err_b1_ols:.2f} |")
print(f"| МНМ    | {beta0_mnm:.3f} | {beta0_mnm - beta0_true:.3f} | {err_b0_mnm:.2f} | {beta1_mnm:.3f} | {beta1_mnm - beta1_true:.3f} | {err_b1_mnm:.2f} |")
print("|        |      |       |        |      |       |        |")
print(f"| МНК (выбросы) | {beta0_ols_out:.3f} | {beta0_ols_out - beta0_true:.3f} | {err_b0_ols_out:.2f} | {beta1_ols_out:.3f} | {beta1_ols_out - beta1_true:.3f} | {err_b1_ols_out:.2f} |")
print(f"| МНМ (выбросы) | {beta0_mnm_out:.3f} | {beta0_mnm_out - beta0_true:.3f} | {err_b0_mnm_out:.2f} | {beta1_mnm_out:.3f} | {beta1_mnm_out - beta1_true:.3f} | {err_b1_mnm_out:.2f} |")