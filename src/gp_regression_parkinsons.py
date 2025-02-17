import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import load_and_preprocess_data

# Initial hyperparameters
theta = 1
r2 = 3
sigma_n2 = 0.001  # Measurement noise

def compute_covariance_matrix(X1, X2, theta, r2, sigma_n2=0):
    """ Compute the Gaussian process covariance matrix """
    N1, _ = X1.shape
    N2, _ = X2.shape
    R = np.zeros((N1, N2))

    for i in range(N1):
        for j in range(N2):
            diff = X1[i] - X2[j]
            R[i, j] = theta * np.exp(-np.dot(diff, diff) / (2 * r2))

    if N1 == N2:  # Add noise term only if it's a square matrix
        R += sigma_n2 * np.eye(N1)

    return R

def gaussian_process_regression(X_train, y_train, X_test, theta, r2, sigma_n2):
    """ Perform Gaussian Process Regression """
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    # Compute covariance matrices
    R_train = compute_covariance_matrix(X_train, X_train, theta, r2, sigma_n2)
    R_cross = compute_covariance_matrix(X_test, X_train, theta, r2)

    # Compute GP estimates
    R_inv = np.linalg.inv(R_train)
    y_pred = R_cross @ R_inv @ y_train

    # Compute variances
    d = compute_covariance_matrix(X_test, X_test, theta, r2, sigma_n2)
    sigma_pred = np.diag(d - R_cross @ R_inv @ R_cross.T)

    return y_pred, np.sqrt(sigma_pred)

def find_nearest_neighbors(X_train, x, N_neighbors=9):
    """ Find the N-1 closest points in the training dataset """
    distances = np.linalg.norm(X_train - x, axis=1)
    nearest_indices = np.argsort(distances)[:N_neighbors]
    return nearest_indices

def evaluate_gp_regression(X_train, y_train, X_val, y_val, theta, r2, sigma_n2):
    """ Perform GP regression for each point in validation set and plot results """
    y_val_pred = np.zeros_like(y_val)

    for k in range(len(X_val)):
        x = X_val[k]
        y_true = y_val[k]

        # Find nearest N-1 neighbors in training set
        nearest_indices = find_nearest_neighbors(X_train, x, N_neighbors=9)
        X_train_subset = X_train[nearest_indices]
        y_train_subset = y_train[nearest_indices]

        # Compute GP estimate
        y_pred, _ = gaussian_process_regression(X_train_subset, y_train_subset, x.reshape(1, -1), theta, r2, sigma_n2)
        y_val_pred[k] = y_pred[0]

    # Plot regression results
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, y_val_pred, c='b', marker='o', label="Predicted vs Actual")
    plt.plot([-2, 2], [-2, 2], 'r--', label="Ideal Fit")
    plt.xlabel("True y_val (Normalized total UPDRS)")
    plt.ylabel("Predicted y_val")
    plt.legend()
    plt.title("Gaussian Process Regression Results")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    # Perform GP regression on validation set
    evaluate_gp_regression(X_train, y_train, X_val, y_val, theta, r2, sigma_n2)
