import numpy as np


# 1. DAMPED OSCILLATOR ODE (second-order converted to first-order)
def damped_oscillator_ode(state, t, m, c, k, external_force=0.0):
    """
    state = [x, v] where:
        x = position
        v = velocity
    External force is included to allow anomaly injection.
    """
    x, v = state
    dxdt = v
    dvdt = (1/m) * (external_force - c*v - k*x)
    return np.array([dxdt, dvdt])



# 2. NUMERICAL SOLVER (RK4)
def rk4_step(func, state, t, dt, *args):
    k1 = func(state, t, *args)
    k2 = func(state + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = func(state + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = func(state + dt*k3, t + dt, *args)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


# 3. SIMULATE TRAJECTORY
def simulate_trajectory(m, c, k, x0, v0, T=20.0, dt=0.001, noise_std=0.01):
    """
    Simulates x(t) with Gaussian measurement noise.
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    state = np.array([x0, v0])
    
    x_arr = np.zeros(n_steps)
    x_arr[0] = x0

    for i in range(1, n_steps):
        state = rk4_step(damped_oscillator_ode, state, t[i], dt, m, c, k)
        x_arr[i] = state[0]

    # Add Gaussian noise
    x_arr += np.random.normal(0, noise_std, size=n_steps)
    
    return t, x_arr


# 4. ANOMALY INJECTION FUNCTIONS
def inject_force_spike(x, spike_t, magnitude=1.0, width=5):
    x_new = x.copy()
    x_new[spike_t:spike_t+width] += magnitude
    return x_new

def inject_damping_change(x, t_idx, new_damping_factor=0.7):
    """
    Approximate damping change by scaling envelope.
    """
    x_new = x.copy()
    x_new[t_idx:] *= new_damping_factor
    return x_new

def inject_freq_shift(x, t_idx, shift_factor=1.2):
    """
    Multiply the phase progression by a factor.
    """
    x_new = x.copy()
    segment = x[t_idx:]
    # brutal frequency shift: compress/stretch time
    new_len = int(len(segment) / shift_factor)
    segment_shifted = np.interp(np.linspace(0, len(segment), new_len),
                                np.arange(len(segment)), segment)
    x_new[t_idx:t_idx+new_len] = segment_shifted
    return x_new

def inject_burst_noise(x, t_idx, duration=20, std=0.3):
    x_new = x.copy()
    x_new[t_idx:t_idx+duration] += np.random.normal(0, std, duration)
    return x_new


# 5. WINDOWING & LABELING
def create_windows(x, window_size, stride=None, anomaly_intervals=None):
    """
    anomaly_intervals: list of (start_idx, end_idx) of injected anomalies.
    Returns:
        X_windows: (num_windows, window_size)
        y_labels:  (num_windows,)
    """
    if stride is None:
        stride = window_size  # non-overlapping
    
    X_windows = []
    y_labels = []

    for start in range(0, len(x) - window_size, stride):
        end = start + window_size
        window = x[start:end]

        # Label = 1 if anomaly interval overlaps window
        label = 0
        if anomaly_intervals is not None:
            for (a_start, a_end) in anomaly_intervals:
                if not (end < a_start or start > a_end):
                    label = 1
                    break
        
        X_windows.append(window)
        y_labels.append(label)
    
    return np.array(X_windows), np.array(y_labels)


# 6. MAIN DATASET GENERATOR
def generate_dataset(num_trajectories=5, save_path="dataset.npz"):
    """
    Generates trajectories, injects anomalies, creates windowed dataset.
    """
    all_X = []
    all_y = []

    for _ in range(num_trajectories):
        # Random physical parameters
        m = np.random.uniform(0.8, 1.2)
        c = np.random.uniform(0.1, 0.4)
        k = np.random.uniform(4.0, 8.0)
        x0 = np.random.uniform(-1.0, 1.0)
        v0 = np.random.uniform(-0.5, 0.5)

        t, x = simulate_trajectory(m, c, k, x0, v0)

        # Inject anomalies
        anomaly_intervals = []
        x_ano = x.copy()

        # 1. Force spike
        idx = np.random.randint(2000, 6000)
        x_ano = inject_force_spike(x_ano, idx, magnitude=0.5)
        anomaly_intervals.append((idx, idx+5))

        # 2. Damping change
        idx2 = np.random.randint(3000, 7000)
        x_ano = inject_damping_change(x_ano, idx2)
        anomaly_intervals.append((idx2, len(x_ano)))

        # 3. Burst noise
        idx3 = np.random.randint(1500, 6500)
        x_ano = inject_burst_noise(x_ano, idx3)
        anomaly_intervals.append((idx3, idx3+20))

        # Windowing (use ~3 periods; pick window_size ~1000 samples)
        window_size = 1000
        Xw, Yw = create_windows(x_ano, window_size, anomaly_intervals=anomaly_intervals)

        all_X.append(Xw)
        all_y.append(Yw)

    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)

    np.savez(save_path, X=X_final, y=y_final)
    print(f"Saved dataset to {save_path}. Shape: {X_final.shape}, Labels: {y_final.shape}")