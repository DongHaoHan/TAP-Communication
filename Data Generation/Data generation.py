import time
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Parameters
# ==========================================
NUM_DATA = 180000
PM_NUM_MIN = 1
PM_NUM_MAX = 5

H_DIPOLE = 0.001
H_SCAN = 0.010
H_SCAN_Target = np.arange(0.005, 0.017, 0.002)

FREQ_SCAN = 3e9
FREQ_Target = np.array([2e9, 3e9, 4e9])

MM, NN = 41, 41
SCAN_STEP = 0.005
SCAN_POINTS = MM * NN

C = 2.99792458e8
MU = 4 * np.pi * 1e-7
EPS_0 = 1e-9 / (36 * np.pi)
WAVE_IMP = np.sqrt(MU / EPS_0)

MAX_MOMENT_P = WAVE_IMP
MIN_MOMENT_P = -MAX_MOMENT_P
MAX_MOMENT_M = 1
MIN_MOMENT_M = -MAX_MOMENT_M

# ==========================================
# 2. Grid & Coordinate Initialization
# ==========================================
x_axis = np.linspace(-SCAN_STEP * (MM - 1) / 2, SCAN_STEP * (MM - 1) / 2, MM)
y_axis = np.linspace(-SCAN_STEP * (NN - 1) / 2, SCAN_STEP * (NN - 1) / 2, NN)
X_grid, Y_grid = np.meshgrid(x_axis, y_axis)

X_flat = X_grid.flatten()
Y_flat = Y_grid.flatten()
X_min, X_max = X_grid.min(), X_grid.max()
Y_min, Y_max = Y_grid.min(), Y_grid.max()

# ==========================================
# 3. Data Storage Initialization
# ==========================================
data_input = np.zeros((NUM_DATA, 1, MM, NN))
data_target = np.zeros((NUM_DATA, 1, MM, NN))

labels_h = np.zeros((NUM_DATA,), dtype=int)
labels_f = np.zeros((NUM_DATA,), dtype=int)

test_set_dict = {}
for f_val in [2, 3, 4]:
    for h_val in range(5, 17, 2):
        test_set_dict[f"{f_val}G_{h_val}mm"] = np.zeros((NUM_DATA, 1, MM, NN))


# ==========================================
# 4. Field Calculation
# ==========================================
def compute_field(h_scan, freq, X_rel, Y_rel, dipoles_vec):
    k = 2 * np.pi * freq / C
    z1 = h_scan - H_DIPOLE
    z2 = h_scan + H_DIPOLE

    r1 = np.sqrt(X_rel ** 2 + Y_rel ** 2 + z1 ** 2)
    r2 = np.sqrt(X_rel ** 2 + Y_rel ** 2 + z2 ** 2)

    fr1 = np.exp(-1j * k * r1) / r1
    fr2 = np.exp(-1j * k * r2) / r2

    g1r1 = (3 / (k * r1) ** 2 + 1j * 3 / (k * r1) - 1) * fr1
    g1r2 = (3 / (k * r2) ** 2 + 1j * 3 / (k * r2) - 1) * fr2

    g2r1 = (2 / (k * r1) ** 2 + 1j * 2 / (k * r1)) * fr1
    g2r2 = (2 / (k * r2) ** 2 + 1j * 2 / (k * r2)) * fr2

    g3r1 = (1 / (k * r1) + 1j) * fr1
    g3r2 = (1 / (k * r2) + 1j) * fr2

    HxMx = (k ** 2 / (4 * np.pi)) * (
            -(Y_rel ** 2 + z1 ** 2) / r1 ** 2 * g1r1 + g2r1
            - (Y_rel ** 2 + z2 ** 2) / r2 ** 2 * g1r2 + g2r2
    )
    HxMy = (k ** 2 / (4 * np.pi)) * (
            X_rel * Y_rel / r1 ** 2 * g1r1 +
            X_rel * Y_rel / r2 ** 2 * g1r2
    )
    HxPz = (k / (4 * np.pi)) * (
            -Y_rel / r1 * g3r1 - Y_rel / r2 * g3r2
    )

    T_matrix = np.concatenate((HxMx, HxMy, HxPz), axis=1)
    return np.dot(T_matrix, dipoles_vec).reshape(MM, NN)


# ==========================================
# 5. Main Generation Loop
# ==========================================
tic = time.time()
print(f"Starting generation for {NUM_DATA} samples...")

for idx in range(NUM_DATA):
    if (idx + 1) % 1000 == 0:
        print(f"Processing {idx + 1}/{NUM_DATA} samples...")

    num_dipoles = np.random.randint(PM_NUM_MIN, PM_NUM_MAX + 1)

    x_pos = (X_min + (X_max - X_min) * np.random.rand(num_dipoles, 1)) * 0.5
    y_pos = (Y_min + (Y_max - Y_min) * np.random.rand(num_dipoles, 1)) * 0.5

    pz_r = MIN_MOMENT_P + (MAX_MOMENT_P - MIN_MOMENT_P) * np.random.rand(num_dipoles, 1)
    pz_i = MIN_MOMENT_P + (MAX_MOMENT_P - MIN_MOMENT_P) * np.random.rand(num_dipoles, 1)
    mx_r = MIN_MOMENT_M + (MAX_MOMENT_M - MIN_MOMENT_M) * np.random.rand(num_dipoles, 1)
    mx_i = MIN_MOMENT_M + (MAX_MOMENT_M - MIN_MOMENT_M) * np.random.rand(num_dipoles, 1)
    my_r = MIN_MOMENT_M + (MAX_MOMENT_M - MIN_MOMENT_M) * np.random.rand(num_dipoles, 1)
    my_i = MIN_MOMENT_M + (MAX_MOMENT_M - MIN_MOMENT_M) * np.random.rand(num_dipoles, 1)

    x_dip_tiled = np.tile(x_pos.T, (SCAN_POINTS, 1))
    y_dip_tiled = np.tile(y_pos.T, (SCAN_POINTS, 1))

    X_rel = np.tile(X_flat, (num_dipoles, 1)).T - x_dip_tiled
    Y_rel = np.tile(Y_flat, (num_dipoles, 1)).T - y_dip_tiled

    Mx = (mx_r + 1j * mx_i).reshape(-1, 1)
    My = (my_r + 1j * my_i).reshape(-1, 1)
    Pz = (pz_r + 1j * pz_i).reshape(-1, 1)
    dipoles_vec = np.concatenate([Mx, My, Pz], axis=0)

    hx_input = compute_field(H_SCAN, FREQ_SCAN, X_rel, Y_rel, dipoles_vec)
    max_val = np.max(np.abs(hx_input))  # Normalization Base
    data_input[idx, 0, :, :] = np.abs(hx_input / max_val)

    h_target = np.random.choice(H_SCAN_Target)
    f_target = np.random.choice(FREQ_Target)

    labels_h[idx] = np.where(np.isclose(H_SCAN_Target, h_target))[0][0]
    labels_f[idx] = np.where(FREQ_Target == f_target)[0][0]

    hx_target = compute_field(h_target, f_target, X_rel, Y_rel, dipoles_vec)
    data_target[idx, 0, :, :] = np.abs(hx_target / max_val)

    if idx >= int(0.99 * NUM_DATA):
        for f_val in FREQ_Target:
            for h_mm in range(5, 17, 2):
                hx_test = compute_field(h_mm / 1e3, f_val, X_rel, Y_rel, dipoles_vec)
                key = f"{int(f_val / 1e9)}G_{h_mm}mm"
                test_set_dict[key][idx, 0, :, :] = np.abs(hx_test / max_val)

# ==========================================
# 6. Save Data to HDF5
# ==========================================
print("Saving dataset to HDF5...")
with h5py.File('Data.h5', 'w') as hf:
    hf.create_dataset('Magnitude_3G_10mm', data=data_input)
    hf.create_dataset('Magnitude_pre', data=data_target)
    hf.create_dataset('Labels_height', data=labels_h)
    hf.create_dataset('Labels_fre', data=labels_f)
    for key, val in test_set_dict.items():
        hf.create_dataset(key, data=val)

print(f"Data saved. Total time: {time.time() - tic:.2f}s")

# ==========================================
# 7. Visualization
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(data_input[0, 0, :, :],
                     extent=(X_min * 1e3, X_max * 1e3, Y_min * 1e3, Y_max * 1e3),
                     origin='upper', aspect='auto', cmap='jet', interpolation='bicubic')
axes[0].set_title('Input: 3GHz @ 10mm')
axes[0].set_xlabel('X (mm)')
axes[0].set_ylabel('Y (mm)')
plt.colorbar(im1, ax=axes[0])

sample_f = FREQ_Target[labels_f[0]] / 1e9
sample_h = H_SCAN_Target[labels_h[0]] * 1e3

im2 = axes[1].imshow(data_target[0, 0, :, :],
                     extent=(X_min * 1e3, X_max * 1e3, Y_min * 1e3, Y_max * 1e3),
                     origin='upper', aspect='auto', cmap='jet', interpolation='bicubic')
axes[1].set_title(f'Target: {int(sample_f)}GHz @ {int(sample_h)}mm')
axes[1].set_xlabel('X (mm)')
axes[1].set_ylabel('Y (mm)')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
