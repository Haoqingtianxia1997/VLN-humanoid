import numpy as np
import matplotlib.pyplot as plt

# vx
average_vx = np.array([0.970, 0.875, 0.783, 0.686, 0.594, 0.499, 0.405, 0.309, 0.219, 0.124, 
                        -0.067, -0.164, -0.257, -0.356, -0.451, -0.547, -0.641, -0.736, -0.833, -0.924])
target_vx = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 
                        -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0])
# vy
average_vy = np.array([0.966, 0.873, 0.781, 0.686, 0.591, 0.496, 0.398, 0.303, 0.205, 0.109, 
                        -0.089, -0.188, -0.286, -0.383, -0.480, -0.572, -0.670, -0.759, -0.852, -0.943])
target_vy = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 
                        -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0])
# wz
average_wz = np.array([2.460, 2.223, 1.978, 1.729, 1.480, 1.246, 1.003, 0.757, 0.521, 0.278,
                       -0.191, -0.414, -0.633, -0.838, -1.019, -1.202, -1.389, -1.554, -1.690, -1.820])
target_wz = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                        -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0])




# choose variable to fit
average = average_vy
target = target_vy

# bias
X = np.vstack([average, np.ones_like(average)]).T

# least squares fit
W, _, _, _ = np.linalg.lstsq(X, target, rcond=None)

slope = W[0]
intercept = W[1]

print(f"target ≈ {slope:.4f} * average + {intercept:.4f}")

fitted = X @ W

plt.scatter(average, target, label="Original Data")
plt.plot(average, fitted, label="Fitted Line", linewidth=2)
plt.xlabel("Average vy")
plt.ylabel("Target vy")
plt.title("Least Squares Linear Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
