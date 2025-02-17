import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import load_and_preprocess_data
from gp_regression_parkinsons import find_nearest_neighbors, gaussian_process_regression

# Define the grid for hyperparameters
r2_values = np.linspace(0.1, 5, 10)  # Try values between 0.1 and 5
sigma_n2_values = np.linspace(0.0001, 0.01, 10)  # Try small values for noise

def compute_validation_mse(X_train, y_train, X_val, y_val, r2, sigma_n2):
    """ Compute validation MSE for given r2 and sigma_n2 """
    y_val_pred = np.zeros_like(y_val)

    for k in range(len(X_val)):
        x = X_val[k]
        nearest_indices = find_nearest_neighbors(X_train, x, N_neighbors=9)
        X_train_subset = X_train[nearest_indices]
        y_train_subset = y_train[nearest_indices]

        y_pred, _ = gaussian_process_regression(X_train_subset, y_train_subset, x.reshape(1, -1), 1, r2, sigma_n2)
        y_val_pred[k] = y_pred[0]

    return np.mean((y_val - y_val_pred) ** 2)  # Mean Squared Error (MSE)

def find_best_hyperparameters(X_train, y_train, X_val, y_val):
    """ Perform grid search to find the best r2 and sigma_n2 """
    best_r2 = None
    best_sigma_n2 = None
    best_mse = float("inf")

    for r2 in r2_values:
        for sigma_n2 in sigma_n2_values:
            mse = compute_validation_mse(X_train, y_train, X_val, y_val, r2, sigma_n2)
            if mse < best_mse:
                best_mse = mse
                best_r2 = r2
                best_sigma_n2 = sigma_n2

    print(f"✅ Best Hyperparameters: r2 = {best_r2}, sigma_n2 = {best_sigma_n2}, MSE = {best_mse}")
    return best_r2, best_sigma_n2

def evaluate_test_set(X_train, y_train, X_test, y_test, r2, sigma_n2, mean_y_train, std_y_train):
    """ Evaluate the GP model on the test dataset """
    y_test_pred = np.zeros_like(y_test)
    sigma_test = np.zeros_like(y_test)

    for k in range(len(X_test)):
        x = X_test[k]
        nearest_indices = find_nearest_neighbors(X_train, x, N_neighbors=9)
        X_train_subset = X_train[nearest_indices]
        y_train_subset = y_train[nearest_indices]

        y_pred, sigma = gaussian_process_regression(X_train_subset, y_train_subset, x.reshape(1, -1), 1, r2, sigma_n2)
        y_test_pred[k] = y_pred[0]
        sigma_test[k] = sigma[0]

    # Un-normalize predictions and standard deviations
    y_test_pred = y_test_pred * std_y_train + mean_y_train
    y_test = y_test * std_y_train + mean_y_train
    sigma_test = sigma_test * std_y_train  # Scale sigma back

    # Compute error metrics
    error = y_test - y_test_pred
    mse = np.mean(error ** 2)
    std_dev = np.std(error)
    mean_error = np.mean(error)

    # Compute R^2 Score
    ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_residual = np.sum((y_test - y_test_pred) ** 2)
    r2_score = 1 - (ss_residual / ss_total)

    print(f"✅ Test Set Evaluation:")
    print(f"Mean Error: {mean_error:.4f}, Std Dev: {std_dev:.4f}, MSE: {mse:.4f}, R²: {r2_score:.4f}")

    # Histogram of errors
    plt.figure(figsize=(8, 6))
    plt.hist(error, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Errors")
    plt.grid()
    plt.show()

    # Estimated vs. True values with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(y_test, y_test_pred, yerr=3 * sigma_test, fmt='o', color='blue', label="Predictions with Error Bars")
    plt.plot(y_test, y_test, 'r--', label="Ideal Fit")
    plt.xlabel("True Total UPDRS")
    plt.ylabel("Predicted Total UPDRS")
    plt.legend()
    plt.title("Test Set Predictions vs. True Values")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    # Get training mean/std for un-normalization
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    # Find optimal hyperparameters
    best_r2, best_sigma_n2 = find_best_hyperparameters(X_train, y_train, X_val, y_val)

    # Evaluate test set with optimized parameters
    evaluate_test_set(X_train, y_train, X_test, y_test, best_r2, best_sigma_n2, mean_y_train, std_y_train)
