import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

MODEL_PATH = "model/bilstm_model.pth"
DEVICE = torch.device("cpu")

SR = 16000
N_FFT = 2048
HOP_LENGTH = 256
NOISE_PROFILE_SEC = 1.0
SUBTRACTION_FACTOR = 2.5
FLOOR = 0.02
VAD_THRESHOLD = 0.015

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
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
    raise FileNotFoundError("Model not found. Train model first!")

def highpass(data, cutoff=120):
    b, a = butter(4, cutoff / (SR / 2), btype='high')
    return lfilter(b, a, data)

def enhance_audio(input_path, output_path):
    audio, _ = librosa.load(input_path, sr=SR)
    audio = highpass(audio)
    
    # Convert full audio to tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
    
    with torch.no_grad():
        enhanced_tensor = model(audio_tensor).cpu().squeeze().numpy()
    
    # Use original length, no truncation
    stft = librosa.stft(enhanced_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = np.abs(stft), np.angle(stft)
    
    noise_frames = int(NOISE_PROFILE_SEC * SR / HOP_LENGTH)
    if noise_frames > mag.shape[1]:
        noise_frames = mag.shape[1] // 2
    noise_profile = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    
    clean_mag = mag - SUBTRACTION_FACTOR * noise_profile
    clean_mag = np.maximum(clean_mag, FLOOR * noise_profile)
    
    energy = np.mean(clean_mag, axis=0)
    voice_mask = energy > VAD_THRESHOLD
    clean_mag[:, ~voice_mask] = 0
    
    clean_audio = librosa.istft(clean_mag * np.exp(1j * phase), hop_length=HOP_LENGTH)
    clean_audio = clean_audio / np.max(np.abs(clean_audio)) * 0.95
    
    sf.write(output_path, clean_audio, SR)