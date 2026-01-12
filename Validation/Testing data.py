import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import DataLoader, TensorDataset
from Model import JHFE_Net

# ==========================================
# 1. Configuration & Parameters
# ==========================================
SAMPLE_INDEX = 0
FONT_SIZE = 45
FREQ_LIST = [2, 3, 4]
HEIGHT_LIST = range(5, 17, 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Trained JHFE-Net.pth"
DATA_PATH = 'Data.h5'

# ==========================================
# 2. Relative Error Function
# ==========================================
def Calculate_relative_error(pre, tru):
    pre = np.abs(pre)
    tru = np.abs(tru)
    error = np.sqrt(np.sum((pre - tru) ** 2) / np.sum(tru ** 2))
    return error

# ==========================================
# 3. Environment & Plotting Style
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 4. Model Loading & Data Initialization
# ==========================================
model = JHFE_Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

with h5py.File(DATA_PATH, 'r') as file:
    dataset_input_all = np.array(file['Magnitude_3G_10mm'])
    num_data = dataset_input_all.shape[0]
    val_end = int(0.99 * num_data)
    test_input = dataset_input_all[val_end:]

    magnitudes_ref = {}
    for key in file.keys():
        if 'G_' in key and 'mm' in key and not key.startswith('Magnitude'):
            magnitudes_ref[key] = np.array(file[key])[val_end:]

test_input_tensor = torch.tensor(test_input, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(test_input_tensor), batch_size=1, shuffle=False)

# ==========================================
# 5. Main Verification & Visualization Loop
# ==========================================
print(f"Starting Verification for Sample Index: {SAMPLE_INDEX}...")

with torch.no_grad():
    found = False
    for i, (test_input_batch,) in enumerate(test_loader):
        if i == SAMPLE_INDEX:
            found = True
            test_input_gpu = test_input_batch.to(DEVICE)

            for f_idx, f_val in enumerate(FREQ_LIST):
                for h_idx, h_val in enumerate(HEIGHT_LIST):
                    label_f = torch.tensor([f_idx], dtype=torch.long).to(DEVICE)
                    label_h = torch.tensor([h_idx], dtype=torch.long).to(DEVICE)

                    output = model(test_input_gpu, label_h, label_f)
                    pred_mag = output.cpu().detach().numpy().squeeze()

                    key = f"{f_val}G_{h_val}mm"
                    ref_mag = magnitudes_ref[key][i, 0, :, :]
                    err = Calculate_relative_error(pred_mag, ref_mag)
                    print(f"Sample {i} | Combination [{key}] | Relative Error: {err:.6f}")

                    plt.figure(figsize=(10, 8))
                    plt.imshow(ref_mag,
                               extent=(-100, 100, -100, 100),
                               origin='upper',
                               aspect='auto',
                               cmap='jet',
                               interpolation='bicubic')
                    plt.colorbar()
                    plt.xlabel('X Axis (mm)')
                    plt.ylabel('Y Axis (mm)')
                    plt.title(r'Reference $|H_x|$')
                    plt.tight_layout()
                    plt.savefig(f'Reference_{key}.png', dpi=300)
                    plt.show()
                    plt.close()

                    plt.figure(figsize=(10, 8))
                    plt.imshow(pred_mag,
                               extent=(-100, 100, -100, 100),
                               origin='upper',
                               aspect='auto',
                               cmap='jet',
                               interpolation='bicubic')
                    plt.colorbar()
                    plt.xlabel('X Axis (mm)')
                    plt.ylabel('Y Axis (mm)')
                    plt.title(r'Predicted $|H_x|$')
                    plt.tight_layout()
                    plt.savefig(f'Pred_{key}.png', dpi=300)
                    plt.show()
                    plt.close()
            break

    if not found:
        print(f"Error: Sample index {SAMPLE_INDEX} is out of range.")


print("\nVerification finished.")
