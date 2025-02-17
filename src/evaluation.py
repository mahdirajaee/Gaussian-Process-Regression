import numpy as np
from covariance_matrix import compute_covariance_matrix
from generate_synthetic import generate_synthetic_data

def select_training_points(t, y, M=10):
    """ Randomly select M training points and one test point t_star """
    t_sampled = np.random.choice(t, M, replace=False)
    t_sampled = np.array(t_sampled, dtype=int)  # Convert to integer indices
    y_sampled = y[t_sampled]  # Extract sampled y values

    # Select a new test point t_star (not in the sampled set)
    t_rem = np.setdiff1d(t, t_sampled)  # Remaining available time points
    t_star = np.random.choice(t_rem, 1, replace=False)[0]
    t_star = int(t_star)  # Ensure it's an integer
    y_true = y[t_star]  # True y value at t_star

    return t_sampled, y_sampled, t_star, y_true

def gaussian_process_regression(t_sampled, y_sampled, t_star, R, sigma_n=0):
    """ Compute the GP regression estimate y_hat(t_star) """
    M = len(t_sampled)
    R_sampled = R[np.ix_(t_sampled, t_sampled)]  # Extract covariance submatrix
    k_star = R[np.ix_([t_star], t_sampled)].T  # Cross covariance

    # Compute GP estimate
    y_hat_star = (k_star.T @ np.linalg.inv(R_sampled + sigma_n * np.eye(M)) @ y_sampled).item()
    sigma = (R[t_star, t_star] - k_star.T @ np.linalg.inv(R_sampled) @ k_star).item()

    return y_hat_star, np.sqrt(sigma)

if __name__ == "__main__":
    t, y, h, t_h = generate_synthetic_data()
    R = compute_covariance_matrix(t)

    t_sampled, y_sampled, t_star, y_true = select_training_points(t, y)

    y_hat, sigma = gaussian_process_regression(t_sampled, y_sampled, t_star, R)
    print(f"True y(t*): {y_true}, Estimated ŷ(t*): {y_hat}, Std Dev: {sigma}")
