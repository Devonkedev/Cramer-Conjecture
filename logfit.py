import math
import matplotlib.pyplot as plt
import pandas as pd
from sympy import primerange
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from parse_primes import parse_all_primes

primes = parse_all_primes('primes_list', num_files=25)

# n_up = 10 ** 9
# n_down = 10 ** 6
# primes = list(primerange(n_down, n_up))
# primes = [p for p in all_primes if n_down <= p < n_up]

def non_consecutive_gaps(primes, k, save=True):
    p1 = primes[:-k]
    p2 = primes[k:]
    gaps = [p2[i] - p1[i] for i in range(len(p1))]
    log_sq = [math.log(x)**2 for x in p1]
    ratio = [gaps[i] / log_sq[i] for i in range(len(gaps))]

    if save:
        df = pd.DataFrame({
            'Prime': p1,
            'Next Prime': p2,
            'Prime Gap': gaps,
            '(log p)^2': log_sq,
            'Gap / (log p)^2': ratio
        })

        # df.to_csv("Cramer.csv", index=False)
    
    return pd.DataFrame({
        "Prime": p1,
        f"Gap_k / (log p)^2": ratio
    })

# k_values = [i for i in range(1, 78000, 10000)]
k_values = [10, 100, 1000, 10000]
non_consec_results = {k: non_consecutive_gaps(primes, k) for k in k_values}


# def log_law(x, A, B):
#     return A / np.log(x) + B

# plt.figure(figsize=(10,7))

# for k in k_values:
#     df_k = non_consec_results[k]
#     x = df_k["Prime"].values
#     y = df_k[f"Gap_k / (log p)^2"].values

#     mask = (x > 10) & (y > 0)
#     x = x[mask]
#     y = y[mask]

#     popt_log, _ = curve_fit(log_law, x, y, maxfev=5000)
#     y_log = log_law(x, *popt_log)

#     plt.scatter(x, y, s=2, alpha=0.4, label=f"data k={k}")
#     plt.plot(x, y_log, lw=2, linestyle="--", label=f"log-law fit k={k}")

# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Prime pₙ")
# plt.ylabel("Gapₖ / (log pₙ)²")
# plt.title("Log-Law Fit to Normalized Non-Consecutive Gaps")
# plt.grid(True, which="both", ls=":")
# plt.legend()
# plt.tight_layout()
# plt.show()

def log_series(x, A, B, C):
    L = np.log(x)
    return A/L + B/(L**2) + C

plt.figure(figsize=(10,7))

for k in k_values:
    df_k = non_consec_results[k]
    x = df_k["Prime"].values
    y = df_k[f"Gap_k / (log p)^2"].values

    mask = (x > 50) & (y > 0)
    x = x[mask]
    y = y[mask]

    popt, _ = curve_fit(log_series, x, y, maxfev=10000)
    A, B, C = popt
    y_fit = log_series(x, *popt)

    r, _ = pearsonr(y, y_fit)
    r2 = r**2

    print(f"\nk={k}")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")
    print(f"R² = {r2:.6f}")

    plt.scatter(x, y, s=2, alpha=0.4, label=f"data k={k}")
    plt.plot(x, y_fit, lw=2, label=f"log-series fit k={k}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Prime pₙ")
plt.ylabel("Gapₖ / (log pₙ)²")
plt.title("High-Accuracy Log-Series Fit")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.show()


# def model2(x, A, B, C, D):
#     L = np.log(x)
#     return A/L + B/(L**2) + C/(L**3) + D

# plt.figure(figsize=(10,7))

# for k in k_values:
#     df_k = non_consec_results[k]
#     x = df_k["Prime"].values
#     y = df_k[f"Gap_k / (log p)^2"].values

#     mask = (x > 50) & (y > 0)
#     x = x[mask]
#     y = y[mask]

#     popt2, _ = curve_fit(model2, x, y, maxfev=20000)
#     y_fit2 = model2(x, *popt2)

#     r, _ = pearsonr(y, y_fit2)
#     r2 = r**2
#     print(f"k={k} → R² = {r2:.6f}")

#     plt.scatter(x, y, s=2, alpha=0.3, label=f"data k={k}")
#     plt.plot(x, y_fit2, lw=2, linestyle="--", label=f"model2 fit k={k}")

# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Prime pₙ")
# plt.ylabel("Gapₖ / (log pₙ)²")
# plt.title("Model 2 Fit: A/log + B/log² + C/log³ + D")
# plt.grid(True, which="both", ls=":")
# plt.legend()
# plt.tight_layout()
# plt.show()
