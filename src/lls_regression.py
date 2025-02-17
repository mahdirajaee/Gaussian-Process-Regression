import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force Matplotlib to use TkAgg (works better on macOS)
import matplotlib.pyplot as plt

def linear_least_squares(t_sampled, y_sampled, t_star):
    """ Compute LLS estimate of y_hat(t_star) """
    A = np.vstack([t_sampled, np.ones(len(t_sampled))]).T  # Linear model: y = ax + b
    a, b = np.linalg.lstsq(A, y_sampled, rcond=None)[0]  # Solve for coefficients

    y_hat_star = a * t_star + b  # Prediction

    return y_hat_star

def generate_synthetic_data():
    # Placeholder function to generate synthetic data
    t = np.linspace(0, 10, 100)
    y = 2 * t + 1 + np.random.normal(0, 1, t.shape)
    h = np.random.normal(0, 1, t.shape)
    t_h = t + h
    return t, y, h, t_h

def select_training_points(t, y):
    # Placeholder function to select training points
    n = len(t)
    indices = np.random.choice(n, size=n//2, replace=False)
    t_sampled = t[indices]
    y_sampled = y[indices]
    t_star = np.linspace(min(t), max(t), 100)
    y_true = 2 * t_star + 1  # Assuming the true model is known
    return t_sampled, y_sampled, t_star, y_true

if __name__ == "__main__":
    t, y, h, t_h = generate_synthetic_data()
    t_sampled, y_sampled, t_star, y_true = select_training_points(t, y)

    y_hat_lls = linear_least_squares(t_sampled, y_sampled, t_star)
    print(f"LLS Estimated ŷ(t*): {y_hat_lls}")
