"""Empirical predictability and information redundancy analysis across representations."""

import sys
from pathlib import Path
import importlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ds = importlib.import_module("06_training.dataset")
generate_transition_dataset = _ds.generate_transition_dataset

def linear_regression_eval(X_train, Y_train, X_test, Y_test, alpha=1e-3):
    X_tr = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_te = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    reg = alpha * np.eye(X_tr.shape[1])
    reg[-1, -1] = 0.0
    W = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ Y_train)
    Y_pred = X_te @ W
    mse = float(np.mean((Y_test - Y_pred) ** 2))
    var_total = float(np.mean(np.var(Y_test, axis=0)))
    r2 = max(0.0, 1.0 - (mse / max(var_total, 1e-8)))
    unexplained = (1.0 - r2) * 100.0
    return mse, r2, unexplained

def run_state_representation_analysis():
    print("=================================================================")
    print(" RELATIONSHIP INFORMATION ANALYSIS: Linear Predictability")
    print("=================================================================")

    data = generate_transition_dataset(n_samples=600, seed=42)

    n_train = 480
    K_flat = data.K_pred.reshape(data.K_pred.shape[0], -1)   # (N, 63)
    G_flat = data.G_pred.reshape(data.G_pred.shape[0], -1)   # (N, 2334)
    R_flat = data.R_pred                                     # (N, 166)
    target_dq = data.delta_q                                 # (N, 21)

    G_mean = np.mean(G_flat[:n_train], axis=0)
    _, _, Vh_G = np.linalg.svd(G_flat[:n_train] - G_mean, full_matrices=False)
    V_g20 = Vh_G[:20].T
    G_comp = (G_flat - G_mean) @ V_g20

    K_tr, K_te = K_flat[:n_train], K_flat[n_train:]
    G_tr, G_te = G_comp[:n_train], G_comp[n_train:]
    R_tr, R_te = R_flat[:n_train], R_flat[n_train:]
    Y_tr, Y_te = target_dq[:n_train], target_dq[n_train:]

    print("\n1. CAN RELATIONSHIP (R) BE RECONSTRUCTED FROM SKELETON / GLOVE?")
    print(f"{'Input Representation':<25} | {'MSE':<12} | {'R^2 Score':<12} | {'Unexplained Variance'}")
    print("-" * 68)

    mse_K, r2_K, un_K = linear_regression_eval(K_tr, R_tr, K_te, R_te)
    print(f"{'K (63D Skeleton)':<25} | {mse_K:<12.6f} | {r2_K:<12.4f} | {un_K:5.1f}%")

    mse_G, r2_G, un_G = linear_regression_eval(G_tr, R_tr, G_te, R_te)
    print(f"{'G (20D Compressed Glove)':<25} | {mse_G:<12.6f} | {r2_G:<12.4f} | {un_G:5.1f}%")

    mse_KG, r2_KG, un_KG = linear_regression_eval(np.hstack([K_tr, G_tr]), R_tr, np.hstack([K_te, G_te]), R_te)
    print(f"{'K + G (83D Combined)':<25} | {mse_KG:<12.6f} | {r2_KG:<12.4f} | {un_KG:5.1f}%")

    print("\n2. DOES RELATIONSHIP (R) IMPROVE TRANSITION PREDICTION (Delta q)?")
    print(f"{'Predictor Representation':<25} | {'MSE':<12} | {'R^2 Score':<12} | {'Unexplained Variance'}")
    print("-" * 68)

    mse_y_K, r2_y_K, un_y_K = linear_regression_eval(K_tr, Y_tr, K_te, Y_te)
    print(f"{'K':<25} | {mse_y_K:<12.6f} | {r2_y_K:<12.4f} | {un_y_K:5.1f}%")

    mse_y_KG, r2_y_KG, un_y_KG = linear_regression_eval(np.hstack([K_tr, G_tr]), Y_tr, np.hstack([K_te, G_te]), Y_te)
    print(f"{'K + G_compressed':<25} | {mse_y_KG:<12.6f} | {r2_y_KG:<12.4f} | {un_y_KG:5.1f}%")

    mse_y_KGR, r2_y_KGR, un_y_KGR = linear_regression_eval(np.hstack([K_tr, G_tr, R_tr]), Y_tr, np.hstack([K_te, G_te, R_te]), Y_te)
    print(f"{'K + G_compressed + R':<25} | {mse_y_KGR:<12.6f} | {r2_y_KGR:<12.4f} | {un_y_KGR:5.1f}%")

    print("\nKey Findings:")
    print(f"- Over {un_KG:.1f}% of R is non-linear geometric context not linearly inferable from K+G.")
    print(f"- Adding R reduces Delta q transition unexplained variance from {un_y_K:.1f}% down to {un_y_KGR:.1f}%.")

if __name__ == "__main__":
    run_state_representation_analysis()
