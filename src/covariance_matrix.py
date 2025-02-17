import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Force Matplotlib to use TkAgg (works better on macOS)
import matplotlib.pyplot as plt

def compute_covariance_matrix(t, T=10):
    """ Compute the covariance matrix using a Gaussian kernel """
    t_new = t[:, np.newaxis]
    delta_t_mat = t_new - t_new.T  # Compute pairwise differences
    R = np.exp(- (delta_t_mat / T) ** 2 / 2)  # Gaussian kernel

    return R

def plot_covariance_matrix(R):
    """ Plot the covariance matrix """
    plt.figure()
    plt.matshow(R, cmap='viridis')
    plt.colorbar()
    plt.title('Theoretical Covariance Matrix')
    plt.show()

def generate_synthetic_data():
    """ Generate synthetic data for testing """
    t = np.linspace(0, 10, 100)
    y = np.sin(t)
    h = np.random.normal(0, 0.1, t.shape)
    t_h = t + h
    return t, y, h, t_h

if __name__ == "__main__":
    t, y, h, t_h = generate_synthetic_data()
    R = compute_covariance_matrix(t)
    plot_covariance_matrix(R)
