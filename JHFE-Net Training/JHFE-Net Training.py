import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from Model import JHFE_Net

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
BATCH_SIZE = 64
LR_INITIAL = 0.001
LR_DECAY = 0.98
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = 'Data.h5'
MODEL_SAVE_PATH = 'Trained_JHFE-Net.pth'
LOSS_PATH = 'Training_Loss_Results.xlsx'

# For evaluation
FREQ_VALUES = [2e9, 3e9, 4e9]
HEIGHT_VALUES = np.arange(0.005, 0.017, 0.002)

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
print(f"Loading data from {DATA_PATH}...")
with h5py.File(DATA_PATH, 'r') as f:
    dataset_input_1 = np.array(f['Magnitude_3G_10mm'])
    dataset_input_2 = np.array(f['Labels_height'])
    dataset_input_3 = np.array(f['Labels_fre'])
    dataset_output = np.array(f['Magnitude_pre'])

total_samples = dataset_input_1.shape[0]
idx_val = int(0.9 * total_samples)
idx_test = int(0.99 * total_samples)


# Split datasets
def to_torch(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype)


# Training Set
train_in1 = to_torch(dataset_input_1[:idx_val])
train_in2 = to_torch(dataset_input_2[:idx_val], torch.long)
train_in3 = to_torch(dataset_input_3[:idx_val], torch.long)
train_out = to_torch(dataset_output[:idx_val])

# Validation Set
val_in1 = to_torch(dataset_input_1[idx_val:idx_test])
val_in2 = to_torch(dataset_input_2[idx_val:idx_test], torch.long)
val_in3 = to_torch(dataset_input_3[idx_val:idx_test], torch.long)
val_out = to_torch(dataset_output[idx_val:idx_test])

train_loader = DataLoader(TensorDataset(train_in1, train_in2, train_in3, train_out),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_in1, val_in2, val_in3, val_out),
                        batch_size=BATCH_SIZE, shuffle=False)

print(f"Datasets: Train={len(train_in1)}, Val={len(val_in1)}, Test={total_samples - idx_test}")

# ==========================================
# 3. Model, Loss, and Optimizer Setup
# ==========================================
model = JHFE_Net().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR_INITIAL)

# ==========================================
# 4. Training Loop
# ==========================================
train_losses = []
val_losses = []
current_lr = LR_INITIAL

print("Starting training process...")
tic_total = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_tic = time.time()

    model.train()
    running_loss = 0.0
    for b1, b2, b3, targets in train_loader:
        b1, b2, b3, targets = b1.to(DEVICE), b2.to(DEVICE), b3.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(b1, b2, b3)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * b1.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for b1, b2, b3, targets in val_loader:
            b1, b2, b3, targets = b1.to(DEVICE), b2.to(DEVICE), b3.to(DEVICE), targets.to(DEVICE)
            outputs = model(b1, b2, b3)
            val_loss = criterion(outputs, targets)
            running_val_loss += val_loss.item() * b1.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    current_lr *= LR_DECAY
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    epoch_duration = time.time() - epoch_tic
    print(f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] - Train Loss: {epoch_train_loss:.6f}, "
          f"Val Loss: {epoch_val_loss:.6f}, Time: {epoch_duration:.2f}s")

print(f"Total Training Time: {time.time() - tic_total:.2f}s")

# ==========================================
# 5. Results Export & Model Saving
# ==========================================
loss_df = pd.DataFrame({
    'Epoch': range(1, NUM_EPOCHS + 1),
    'Train_Loss': train_losses,
    'Val_Loss': val_losses
})
loss_df.to_excel(LOSS_PATH, index=False)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# ==========================================
# 6. Evaluation on Test Set
# ==========================================
print("\nRunning Final Evaluation...")
model.eval()

with h5py.File(DATA_PATH, 'r') as f:
    test_keys = [k for k in f.keys() if 'G_' in k and 'mm' in k and not k.startswith('Magnitude')]
    test_keys.sort()

    total_mse = 0.0
    for key in test_keys:
        # Parse labels from key
        f_str, h_str = key.split('_')
        f_val = float(f_str[:-1]) * 1e9
        h_val = float(h_str[:-2]) / 1000.0

        f_idx = FREQ_VALUES.index(f_val)
        h_idx = np.where(np.isclose(HEIGHT_VALUES, h_val))[0][0]

        test_in = to_torch(dataset_input_1[idx_test:]).to(DEVICE)
        lbl_h = torch.full((test_in.size(0),), h_idx, dtype=torch.long).to(DEVICE)
        lbl_f = torch.full((test_in.size(0),), f_idx, dtype=torch.long).to(DEVICE)

        test_target = to_torch(np.array(f[key])[idx_test:]).to(DEVICE)

        with torch.no_grad():
            preds = model(test_in, lbl_h, lbl_f)
            mse = criterion(preds, test_target).item()
            total_mse += mse
            print(f"  Target [{key}] -> MSE: {mse:.6f}")

    print(f"Average Test MSE: {total_mse / len(test_keys):.6f}")

# ==========================================
# 7. Visualization
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", linewidth=2)
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("JHFE-Net Training History")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Loss_Curve.png', dpi=300)
plt.show()
