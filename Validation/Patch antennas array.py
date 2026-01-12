import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from Model import JHFE_Net

# ==========================================
# 1. Configuration & Parameters
# ==========================================
ANTENNA_LIST = ['patchs_12', 'patchs_13', 'patchs_1234']
FREQ_LIST = [3]
HEIGHT_LIST = range(5, 17, 2)
GRID_SIZE = 41
MODEL_PATH = 'Trained JHFE-Net.pth'
FONT_SIZE = 45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. Relative Error Function
# ==========================================
def Calculate_relative_error(pre, tru):
    pre = np.abs(pre)
    tru = np.abs(tru)
    error = np.sqrt(np.sum((pre - tru) ** 2) / np.sum(tru ** 2))
    return error

# ==========================================
# 3. Data Load
# ==========================================
def Load_gt_raw(filename):
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        return None
    data = np.loadtxt(filename, comments=['#', '*'])
    re_hx = data[:, 9]
    im_hx = data[:, 10]
    mag_full = np.sqrt(re_hx ** 2 + im_hx ** 2)
    mag_by_height = mag_full.reshape(-1, len(HEIGHT_LIST))
    frames = []
    for h_idx in range(len(HEIGHT_LIST)):
        grid = mag_by_height[:, h_idx].reshape(GRID_SIZE, GRID_SIZE)
        frames.append(grid)
    return frames


# ==========================================
# 4. Environment & Plotting Style
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 5. Model Loading & Initialization
# ==========================================
model = JHFE_Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# ==========================================
# 6. Main Verification & Visualization Loop
# ==========================================

for a_idx, antenna in enumerate(ANTENNA_LIST):
    input_file = f'Scanning_{antenna}_3GHz_10mm.near'

    if not os.path.exists(input_file):
        print(f"Skip: {antenna}, file {input_file} not found.")
        continue

    in_data = np.loadtxt(input_file, comments=['#', '*'])
    in_mag_raw = np.sqrt(in_data[:, 9] ** 2 + in_data[:, 10] ** 2).reshape(GRID_SIZE, GRID_SIZE)
    norm_factor = np.max(in_mag_raw)
    input_tensor = torch.tensor(in_mag_raw / norm_factor, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    for f_val in FREQ_LIST:
        gt_filename = f'scanning_{antenna}_{f_val}GHz.near'
        gt_frames_raw = Load_gt_raw(gt_filename)

        if gt_frames_raw is None:
            continue

        for h_idx, h_val in enumerate(HEIGHT_LIST):
            gt_frame = gt_frames_raw[h_idx]
            key = f"{f_val}G_{h_val}mm"

            with torch.no_grad():
                label_f = torch.tensor([1], dtype=torch.long).to(DEVICE)
                label_h = torch.tensor([h_idx], dtype=torch.long).to(DEVICE)
                pred_output = model(input_tensor, label_h, label_f).squeeze().cpu().numpy()
                pred_denorm = pred_output * norm_factor
                err_val = Calculate_relative_error(pred_denorm, gt_frame)

            print(f"[{antenna}_{key}] | Relative Error: {err_val:.6f}")

            plt.figure(figsize=(10, 8))
            plt.imshow(gt_frame,
                       extent=(-100, 100, -100, 100),
                       cmap='jet',
                       origin='upper',
                       interpolation='bicubic',
                       aspect='auto')
            plt.colorbar()
            plt.xlabel('X Axis (mm)')
            plt.ylabel('Y Axis (mm)')
            plt.title(r'Reference $|H_x|$')
            plt.tight_layout()
            plt.savefig(f'Reference_{antenna}_{key}.png', dpi=300)
            plt.close()

            plt.figure(figsize=(10, 8))
            plt.imshow(pred_denorm,
                       extent=(-100, 100, -100, 100),
                       cmap='jet',
                       origin='upper',
                       interpolation='bicubic',
                       aspect='auto')
            plt.colorbar()
            plt.xlabel('X Axis (mm)')
            plt.ylabel('Y Axis (mm)')
            plt.title(r'Predicted $|H_x|$')
            plt.tight_layout()
            plt.savefig(f'Pred_{antenna}_{key}.png', dpi=300)
            plt.close()

print("\nValidation finished.")