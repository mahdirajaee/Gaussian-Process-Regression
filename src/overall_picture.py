import numpy as np
import matplotlib.pyplot as plt
from generate_synthetic import generate_synthetic_data
from covariance_matrix import compute_covariance_matrix
from gp_regression import select_training_points, gaussian_process_regression
from lls_regression import linear_least_squares

def compute_full_gp_regression(t_sampled, y_sampled, t, R):
    """ Compute GP regression for all possible values of t_* """
    y_hat_gp = np.zeros_like(t, dtype=float)
    sigma_gp = np.zeros_like(t, dtype=float)

    for i, t_star in enumerate(t):
        y_hat_gp[i], sigma_gp[i] = gaussian_process_regression(t_sampled, y_sampled, t_star, R)

    return y_hat_gp, sigma_gp

def plot_overall_picture(t, y, t_sampled, y_sampled, y_hat_gp, sigma_gp):
    """ Plot the full GP regression, confidence bounds, and LLS regression """
    plt.figure(figsize=(10, 6))

    # Process realization (red line)
    plt.plot(t, y, color='red', linewidth=2, label="Process realization")

    # Sampled points (known values, red dots)
    plt.scatter(t_sampled, y_sampled, color='red', marker='o', s=60, label="Known values")

    # GP regression estimate (blue line)
    plt.plot(t, y_hat_gp, color='blue', linewidth=2, label="GP regression")

    # Confidence intervals: min/max GP regression
    plt.plot(t, y_hat_gp - sigma_gp, color='blue', linestyle='dashed', label="min of GP regression")
    plt.plot(t, y_hat_gp + sigma_gp, color='blue', linestyle='dotted', label="max of GP regression")

    # LLS regression line (black)
    A = np.vstack([t_sampled, np.ones(len(t_sampled))]).T
    a, b = np.linalg.lstsq(A, y_sampled, rcond=None)[0]
    plt.plot(t, a * t + b, color='black', linewidth=2, label="LLS predicted")

    plt.xlabel("t (s)")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.title("The Overall Picture: GP Regression vs LLS")
    plt.show()

if __name__ == "__main__":
    # Generate synthetic data
    t, y, h, t_h = generate_synthetic_data()
    R = compute_covariance_matrix(t)

    # Select training points
    t_sampled, y_sampled, _, _ = select_training_points(t, y)

    # Compute GP regression for all t_*
    y_hat_gp, sigma_gp = compute_full_gp_regression(t_sampled, y_sampled, t, R)

    # Plot the overall picture
    plot_overall_picture(t, y, t_sampled, y_sampled, y_hat_gp, sigma_gp)
