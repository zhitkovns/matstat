import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

# ---------- Генерация данных ----------
x = np.arange(-1.8, 2.01, 0.2)   # 20 точек
true_a, true_b = 2.0, 2.0
eps = np.random.normal(0, 1, len(x))
y = true_a + true_b * x + eps

# Функция для МНК
def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b

# Функция для МНМ
def lad(x, y):
    def objective(params):
        a, b = params
        return np.sum(np.abs(y - (a + b * x)))
    res = minimize(objective, x0=[0, 0], method='Nelder-Mead')
    return res.x

# ---------- Без выбросов ----------
a_ols, b_ols = ols(x, y)
a_lad, b_lad = lad(x, y)

# Относительные погрешности
def rel_error(true, est):
    return np.abs((true - est) / true) * 100

err_a_ols = rel_error(true_a, a_ols)
err_b_ols = rel_error(true_b, b_ols)
err_a_lad = rel_error(true_a, a_lad)
err_b_lad = rel_error(true_b, b_lad)

print("=== Без выбросов ===")
print(f"МНК: a = {a_ols:.4f}, b = {b_ols:.4f}")
print(f"МНМ: a = {a_lad:.4f}, b = {b_lad:.4f}")

# ---------- С выбросами ----------
y_out = y.copy()
y_out[0] += 10
y_out[-1] -= 10

a_ols_out, b_ols_out = ols(x, y_out)
a_lad_out, b_lad_out = lad(x, y_out)

err_a_ols_out = rel_error(true_a, a_ols_out)
err_b_ols_out = rel_error(true_b, b_ols_out)
err_a_lad_out = rel_error(true_a, a_lad_out)
err_b_lad_out = rel_error(true_b, b_lad_out)

print("\n=== С выбросами ===")
print(f"МНК: a = {a_ols_out:.4f}, b = {b_ols_out:.4f}")
print(f"МНМ: a = {a_lad_out:.4f}, b = {b_lad_out:.4f}")

# ---------- Построение графиков с одинаковыми границами ----------
x_line = np.linspace(-2, 2.2, 100)
y_true = true_a + true_b * x_line
y_ols = a_ols + b_ols * x_line
y_lad = a_lad + b_lad * x_line
y_ols_out = a_ols_out + b_ols_out * x_line
y_lad_out = a_lad_out + b_lad_out * x_line

all_y = np.concatenate([y, y_out, y_true, y_ols, y_lad, y_ols_out, y_lad_out])
x_min, x_max = -2.2, 2.2
y_min, y_max = all_y.min(), all_y.max()
y_pad = (y_max - y_min) * 0.1
y_min -= y_pad
y_max += y_pad

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Без выбросов
ax = axes[0]
ax.scatter(x, y, color='blue', label='Данные')
ax.plot(x_line, y_true, 'k-', label='Истинная прямая')
ax.plot(x_line, y_ols, 'r--', label=f'МНК: a={a_ols:.2f}, b={b_ols:.2f}')
ax.plot(x_line, y_lad, 'g--', label=f'МНМ: a={a_lad:.2f}, b={b_lad:.2f}')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Без выбросов')
ax.legend()
ax.grid(True, alpha=0.3)

# С выбросами
ax = axes[1]
ax.scatter(x, y_out, color='blue', label='Данные с выбросами')
ax.plot(x_line, y_true, 'k-', label='Истинная прямая')
ax.plot(x_line, y_ols_out, 'r--', label=f'МНК: a={a_ols_out:.2f}, b={b_ols_out:.2f}')
ax.plot(x_line, y_lad_out, 'g--', label=f'МНМ: a={a_lad_out:.2f}, b={b_lad_out:.2f}')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('С выбросами')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression.png', dpi=150)
plt.show()

# ---------- Вывод таблицы для отчёта ----------
print("\nРезультаты в виде таблицы:")
print("|        | a   | Δa   | δa, % | b   | Δb   | δb, % |")
print("|--------|-----|------|-------|-----|------|-------|")
print(f"| МНК    | {a_ols:.3f} | {a_ols - true_a:.3f} | {err_a_ols:.2f} | {b_ols:.3f} | {b_ols - true_b:.3f} | {err_b_ols:.2f} |")
print(f"| МНМ    | {a_lad:.3f} | {a_lad - true_a:.3f} | {err_a_lad:.2f} | {b_lad:.3f} | {b_lad - true_b:.3f} | {err_b_lad:.2f} |")
print("|        |     |      |       |     |      |       |")
print(f"| МНК (выбросы) | {a_ols_out:.3f} | {a_ols_out - true_a:.3f} | {err_a_ols_out:.2f} | {b_ols_out:.3f} | {b_ols_out - true_b:.3f} | {err_b_ols_out:.2f} |")
print(f"| МНМ (выбросы) | {a_lad_out:.3f} | {a_lad_out - true_a:.3f} | {err_a_lad_out:.2f} | {b_lad_out:.3f} | {b_lad_out - true_b:.3f} | {err_b_lad_out:.2f} |")