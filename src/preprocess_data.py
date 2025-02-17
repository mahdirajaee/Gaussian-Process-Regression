import numpy as np

def load_and_preprocess_data(filename="data/parkinsons_updrs.csv"):
    """ 
    Load, shuffle, split, and normalize the Parkinson's dataset. 
    Keeps only required columns:
      - Regressand: total_UPDRS
      - Regressors: motor_UPDRS, age, PPE
    """

    # Load dataset (assumes CSV format with headers)
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Select relevant columns:
    # - Column 0: total_UPDRS (regressand)
    # - Columns 1, 2, 3: motor_UPDRS, age, PPE (regressors)
    X = data[:, [1, 2, 3]]  # Extract regressors (Motor UPDRS, Age, PPE)
    y = data[:, 0]          # Extract regressand (total_UPDRS)

    # Shuffle the dataset
    np.random.seed(42)  # Ensure reproducibility
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split into train (50%), validation (25%), test (25%)
    N = X.shape[0]
    N_train = int(0.50 * N)
    N_val = int(0.25 * N)
    N_test = N - N_train - N_val  # Ensure all data is used

    X_train, y_train = X[:N_train], y[:N_train]
    X_val, y_val = X[N_train:N_train + N_val], y[N_train:N_train + N_val]
    X_test, y_test = X[N_train + N_val:], y[N_train + N_val:]

    # Compute mean and standard deviation from training data
    mean_X_train = np.mean(X_train, axis=0)
    std_X_train = np.std(X_train, axis=0)

    # Normalize datasets using training statistics
    X_train_norm = (X_train - mean_X_train) / std_X_train
    X_val_norm = (X_val - mean_X_train) / std_X_train
    X_test_norm = (X_test - mean_X_train) / std_X_train

    # Normalize y (optional)
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    y_train_norm = (y_train - mean_y_train) / std_y_train
    y_val_norm = (y_val - mean_y_train) / std_y_train
    y_test_norm = (y_test - mean_y_train) / std_y_train

    return X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm

if __name__ == "__main__":
    # Run preprocessing and check output
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm = load_and_preprocess_data()

    print("Data successfully preprocessed and normalized.")
    print(f"Shapes: X_train={X_train_norm.shape}, X_val={X_val_norm.shape}, X_test={X_test_norm.shape}")
    print(f"y_train shape: {y_train_norm.shape}, y_val shape: {y_val_norm.shape}, y_test shape: {y_test_norm.shape}")
