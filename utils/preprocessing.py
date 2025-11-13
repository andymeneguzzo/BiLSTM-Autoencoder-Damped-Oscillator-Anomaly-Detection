import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# 1. NORMALIZATION
def fit_normal_scaler(X: np.ndarray, y: np.ndarray) -> StandardScaler:
    """
    Fits StandardScaler ONLY on normal windows.
    X shape: (num_windows, window_len) or (num_windows, window_len, num_features)
    y shape: (num_windows,) containing {0,1}

    We flatten time dims for scaling per-feature.
    """
    normal_idx = np.where(y == 0)[0]
    X_normal = X[normal_idx]

    # If univariate: reshape to (N*window, 1)
    if X_normal.ndim == 2:
        X_flat = X_normal.reshape(-1, 1)
    else:
        # Multivariate: reshape to (N*window, num_features)
        _, T, F = X_normal.shape
        X_flat = X_normal.reshape(-1, F)
    
    scaler = StandardScaler()
    scaler.fit(X_flat)

    return scaler

def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Applies fitted scaler to full dataset.
    """
    if X.ndim == 2:
        # (num_windows, window_len)
        X_flat = X.reshape(-1, 1)
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
    else:
        # (num_windows, window_len, num_features)
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled = scaler.transform(X_flat).reshape(N, T, F)

    return X_scaled

# 2. SPLITTING (TRAIN ONLY NORMAL DATA)
def split_dataset(
        X: np.ndarray, y: np.ndarray,
        train_ratio=0.7, val_ratio=0.15, random_state=0
    ) -> Dict[str, np.ndarray]:
    """
    - Train: only normal windows (y=0)
    - Val/Test: mix of both classes
    - Shuffle before split
    """

    rng = np.random.default_rng(random_state)

    # Shuffle
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Identify normal windows for training
    normal_idx = np.where(y == 0)[0]
    n_train = int(train_ratio * len(normal_idx))
    train_idx = normal_idx[:n_train]

    # Remaining windows (normal + anomalous) → split into val/test
    """
    remaining_idx = np.setdiff1d(np.arange(len(X)), train_idx)
    n_val = int(val_ratio * len(X))
    n_val = min(n_val, len(remaining_idx)//2)
    val_idx = remaining_idx[:n_val]
    test_idx = remaining_idx[n_val:]
    """
    # Fix: ensure test set has anomalies, otherwise could be all in validation and none in test
    remaining_idx = np.setdiff1d(np.arange(len(X)), train_idx)

    # Split remaining into normal vs anomaly
    remaining_norm = remaining_idx[y[remaining_idx] == 0]
    remaining_anom = remaining_idx[y[remaining_idx] == 1]

    # Determine validation size
    n_val_total = int(val_ratio * len(X))

    # Split anomalies first (ensure both val and test get some)
    n_val_anom = max(1, min(len(remaining_anom)//2,  int(0.5 * n_val_total)))
    val_anom_idx = remaining_anom[:n_val_anom]
    test_anom_idx = remaining_anom[n_val_anom:]

    # Fill remaining val slots with normal windows
    n_val_norm = n_val_total - len(val_anom_idx)
    val_norm_idx = remaining_norm[:n_val_norm]
    test_norm_idx = remaining_norm[n_val_norm:]

    # Merge
    val_idx = np.concatenate([val_norm_idx, val_anom_idx])
    test_idx = np.concatenate([test_norm_idx, test_anom_idx])

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx
    }

# 3. SAVE AND LOAD
def save_preprocessed(path: str, data_dict: Dict[str, np.ndarray], scaler: StandardScaler):
    """
    Saves all arrays + scaler parameters into npz.
    """
    np.savez(
        path,
        **data_dict,
        scaler_mean=scaler.mean_,
        scaler_std=scaler.scale_
    )
    print(f"Saved preprocessed dataset to {path}")


def load_preprocessed(path: str) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """
    Loads npz dataset and reconstructs StandardScaler.
    """
    data = np.load(path, allow_pickle=True)

    scaler = StandardScaler()
    scaler.mean_ = data["scaler_mean"]
    scaler.scale_ = data["scaler_std"]

    data_dict = {key: data[key] for key in data.files if key not in ["scaler_mean", "scaler_std"]}

    return data_dict, scaler


# 4. MAIN PREPROCESSING FUNCTION READY TO USE
def preprocess_dataset(raw_path="dataset.npz", save_path="preprocessed_dataset.npz"):
    """
    Full pipeline:
        - Load raw dataset
        - Fit scaler on normal windows
        - Apply scaler to all windows
        - Split into train/val/test
        - Save output
    """
    raw = np.load(raw_path)
    X = raw["X"]
    y = raw["y"]

    # Fit scaler on normal windows only
    scaler = fit_normal_scaler(X, y)

    # Apply scaling
    X_scaled = apply_scaler(X, scaler)

    # Split into train/val/test
    data_split = split_dataset(X_scaled, y)

    # Save everything
    save_preprocessed(save_path, data_split, scaler)

    print("Preprocessing complete.")
    return data_split, scaler


if __name__ == "__main__":
    preprocess_dataset()