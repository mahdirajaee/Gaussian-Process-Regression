# Gaussian Process Regression for Synthetic and Parkinson’s Disease Data

**Author:** *Mahdi Rajaee*  

**Institution:** **Politecnico di Torino**

**Date:** *17/02/2025*

---

## 1. Introduction
Gaussian Process (GP) regression is a powerful non-parametric technique for modeling complex, continuous-valued functions. Unlike traditional linear regression, GP regression places a prior over functions, enabling robust uncertainty quantification and flexible function fitting. This report presents two applications of GP regression:

1. **Synthetic Data Experiment**: Generating and analyzing a Gaussian random process filtered by a Gaussian kernel.  
2. **Parkinson’s Disease Dataset**: Predicting the `total_UPDRS` measure (Unified Parkinson’s Disease Rating Scale) from clinical features.

The primary goals are:
- Demonstrating how to build and use GP regression for a **time-series** (synthetic data).
- Showing how to apply GP regression to **real-world clinical data** (Parkinson’s dataset).
- Exploring **hyperparameter tuning** (length scale \(r\), noise variance \(\sigma_n^2\)) to minimize prediction error and compare with Linear Least Squares (LLS).

---

## 2. Synthetic Data Experiment

### 2.1 Data Generation
We generate a **white Gaussian noise** signal \(x(t)\) of length \(N_p = 200\), then convolve it with a **Gaussian impulse response** \(h(t)\) of finite duration to form a **synthetic Gaussian random process** \(y(t)\). Specifically:

1. **White Noise**:  
   \[
   x \sim \mathcal{N}(0, 1)
   \]
   for \(t \in \{0,1,\ldots,N_p-1\}\).

2. **Gaussian Filter**:  
   \[
   h(t) = \exp\left(- \frac{t^2}{T^2}\right)
   \]
   normalized to unit energy.

3. **Filtered Output**:  
   \[
   y(t) = (x * h)(t),
   \]
   convolved with `mode='same'` so the output matches the original time support.

4. **Impulse Response & Realization**: We visualize:
   - **Impulse response** \(h(t)\) vs. \(t\).  
   - **Process realization** \(y(t)\) vs. \(t\).

### 2.2 Covariance Computation
The filter design ensures \(y(t)\) is a zero-mean **Gaussian process**. Its autocorrelation \(R_Y(\tau)\) is analytically equivalent to a Gaussian:

\[
R_Y(\tau) = \exp\left(-\frac{\tau^2}{2T^2}\right).
\]

From a finite set of points \(\{t_k\}\), we build the covariance matrix \(R\) via:

\[
R_{ij} = R_Y(t_i - t_j).
\]

### 2.3 GP Regression Procedure
1. **Sampling**: Select \(M = 10\) random indices \(\{t_k\}\) as training points \(\{(t_k, y(t_k))\}\).  
2. **Test Point**: Randomly pick another point \(t_*\) for regression.  
3. **Covariance Matrix**: Compute:
   \[
   R_{Y,N} =
   \begin{pmatrix}
   R_{M\times M} & k \\
   k^T & d
   \end{pmatrix},
   \]
   where
   - \(R_{M\times M}\) is the training covariance submatrix,
   - \(k\) is the cross-covariance with the test point,
   - \(d\) is the test point’s variance.  
4. **GP Prediction**:
   \[
   \hat{y}(t_*) = k^T \, R_{M\times M}^{-1} \, \mathbf{y}
   \quad\text{and}\quad
   \sigma^2 = d - k^T \, R_{M\times M}^{-1} \, k.
   \]
5. **Results**:
   - We plot **training points** (markers), **true process** \(y(t)\), and the **GP estimate** at \(t_*\).  
   - Compare with **Linear Least Squares (LLS)** by fitting \(y(t) \approx a\,t + b\).  
   - Display the confidence interval \(\pm \sigma\).

**Outcome**: Repeated experiments confirm GP regression can accurately reconstruct missing points from a small training set, outperforming a simple LLS fit especially when data is distinctly non-linear.

---

## 3. Parkinson’s Disease Dataset

### 3.1 Data Description
The Parkinson’s dataset contains **clinical features** and **UPDRS measures**. For this experiment:
- **Regressand**: `total_UPDRS`
- **Regressors**: `motor_UPDRS`, `age`, `PPE` (3 features)
- Dataset is **shuffled** and **split** into:
  - **Training set** (50%)
  - **Validation set** (25%)
  - **Test set** (25%)
- We **normalize** each feature by subtracting training mean and dividing by training std.

### 3.2 GP Regression Model
Following the same GP framework:
\[
R_{Y}(n,k) = \theta \exp\Bigl(-\frac{\|x_n - x_k\|^2}{2r^2}\Bigr) + \sigma_{\nu}^2 \delta_{n,k}.
\]
Because the data is normalized, we set:
- \(\theta = 1\),
- Hyperparameters to tune: \(r^2\) (length scale) and \(\sigma_{\nu}^2\) (noise variance).

**Implementation Steps**:
1. **Nearest Neighbors**: For computational efficiency, we often pick \(N = 10\) neighbors from the training set closest to a validation point \(x\).  
2. **Build Sub-covariance** among these \(N-1\) neighbors plus the target point.  
3. **Predict** \(\hat{y}\) using the GP formula.  
4. **Evaluate** the MSE on the **validation set**.  
5. **Tune** \(r^2\) and \(\sigma_{\nu}^2\) to minimize validation MSE (grid search).

### 3.3 Hyperparameter Tuning
We **grid search** over a range of plausible \(r^2\) and \(\sigma_{\nu}^2\) values:
- **Range** for \(r^2\): e.g. `[0.1, 0.5, 1.0, 2.0, ..., 5.0]`.
- **Range** for \(\sigma_{\nu}^2\): e.g. `[1e-4, 5e-4, 1e-3, ..., 1e-2]`.
- Compute validation MSE \(\min \|y_\text{val} - y_\text{val_pred}\|^2\).

**Optimal Hyperparameters** are then selected to minimize the validation error.

### 3.4 Test Set Evaluation
With the best \(r^2\) and \(\sigma_{\nu}^2\) found, we:
1. **Predict** `total_UPDRS` for every test sample.
2. **Un-normalize** predictions and compute:
   - **Mean Error, Standard Deviation, MSE** of `y_test - y_hat`.
   - **Histogram** of prediction errors.
   - **\(R^2\) Score** to measure goodness of fit.  
3. **Plot** predicted vs. true unnormalized `total_UPDRS` with \(3\sigma\) error bars.

**Result**: A typical improvement over LLS in capturing non-linearities of the clinical data.

---

## 4. Conclusion
This research demonstrates **Gaussian Process Regression** on both **synthetic** and **real-world** datasets. Key insights:

1. **Synthetic Data**:  
   - GP regression accurately recovers process values using a small subset of points.  
   - **Confidence intervals** reflect true process variability better than linear methods.

2. **Parkinson’s Disease Dataset**:  
   - GP regression with tuned hyperparameters provides a stronger predictive performance compared to basic LLS.  
   - The ability to model **non-linear relationships** between `motor_UPDRS`, `age`, `PPE`, and `total_UPDRS` is crucial in medical applications.

3. **Hyperparameter Tuning**:
   - Proper selection of **length scale \(r\)** and **noise variance \(\sigma_{\nu}^2\)** significantly impacts prediction accuracy.  
   - Using a **validation set** is essential to avoid overfitting.

4. **Comparison with LLS**:
   - While LLS can be a baseline approach, **GP** typically yields **lower MSE** and provides **uncertainty estimates**.

Overall, **Gaussian Process Regression** proves to be a flexible, robust approach for both synthetic and real-world data, and it is highly recommended for tasks where **uncertainty quantification** and **non-linear modeling** are paramount.

