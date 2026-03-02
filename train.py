import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa

CLEAN_DIR = "dataset/clean"
NOISY_DIR = "dataset/noisy"
SR = 16000
FIXED_LEN = SR
DEVICE = torch.device("cpu")
BATCH_SIZE = 8
EPOCHS = 100

class SpeechDataset(Dataset):
    def __init__(self, max_files=50):
        self.files = os.listdir(CLEAN_DIR)[:max_files]
        if len(self.files) == 0:
            raise ValueError("Clean folder is empty")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clean_path = os.path.join(CLEAN_DIR, self.files[idx])
        noisy_path = os.path.join(NOISY_DIR, self.files[idx])
        clean, _ = librosa.load(clean_path, sr=SR)
        noisy, _ = librosa.load(noisy_path, sr=SR)
        if len(clean) > FIXED_LEN:
            clean = clean[:FIXED_LEN]
            noisy = noisy[:FIXED_LEN]
        else:
            clean = librosa.util.fix_length(clean, size=FIXED_LEN)
            noisy = librosa.util.fix_length(noisy, size=FIXED_LEN)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(-1)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(-1)
        return noisy, clean

dataset = SpeechDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

model = BiLSTM().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started on CPU...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/bilstm_model.pth")
print("Training done. Model saved at model/bilstm_model.pth")
