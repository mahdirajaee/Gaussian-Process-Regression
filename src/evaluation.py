import numpy as np
import matplotlib.pyplot as plt
from generate_synthetic import generate_synthetic_data
from covariance_matrix import compute_covariance_matrix
from gp_regression import select_training_points, gaussian_process_regression
from lls_regression import linear_least_squares

def plot_results(t, y, t_sampled, y_sampled, t_star, y_hat_gp, sigma, y_hat_lls, y_true):
    """ Plot realization, GP regression, and LLS regression """
    plt.figure(figsize=(10, 6))

    # Plot full Gaussian Process realization
    plt.plot(t, y, 'b', alpha=0.7, label="realization")

    # Mark sampled points (training data)
    plt.scatter(t_sampled, y_sampled, color='blue', marker='o', s=60, label="sampled values")

    # Mark the true value y(t*)
    plt.scatter(t_star, y_true, color='red', marker='s', s=80, label="true value")

    # Mark GP regression prediction ŷ(t*)
    plt.scatter(t_star, y_hat_gp, color='green', marker='x', s=100, label="GP regression")

    # Plot confidence interval: ŷ(t*) ± σ
    plt.plot([t_star, t_star], [y_hat_gp - sigma, y_hat_gp + sigma], color='green', linewidth=2, label="range of GP regression")

    # LLS regression line (black)
    A = np.vstack([t_sampled, np.ones(len(t_sampled))]).T
    a, b = np.linalg.lstsq(A, y_sampled, rcond=None)[0]
    plt.plot(t, a * t + b, color='black', linewidth=1.5, label="LLS regression")

    # Mark LLS prediction ŷ(t*) as '+'
    plt.scatter(t_star, y_hat_lls, color='black', marker='+', s=100, label="LLS predicted")

    plt.xlabel("t (s)")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.title("Gaussian Process Regression vs Linear Least Squares")
    plt.show()

if __name__ == "__main__":
    # Generate synthetic data
    t, y, h, t_h = generate_synthetic_data()
    R = compute_covariance_matrix(t)

    # Select training points and one test point
    t_sampled, y_sampled, t_star, y_true = select_training_points(t, y)

    # Compute GP regression estimate
    y_hat_gp, sigma = gaussian_process_regression(t_sampled, y_sampled, t_star, R)

    # Compute LLS regression estimate
    y_hat_lls = linear_least_squares(t_sampled, y_sampled, t_star)

    # Plot results
    plot_results(t, y, t_sampled, y_sampled, t_star, y_hat_gp, sigma, y_hat_lls, y_true)
