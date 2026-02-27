import sounddevice as sd
import soundfile as sf
import os

SR = 16000
DURATION = 5
OUTPUT_PATH = "test_audio/noisy.wav"

def record_audio():
    os.makedirs("test_audio", exist_ok=True)
    print("Recording for 5 seconds...")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    sf.write(OUTPUT_PATH, audio, SR)
    print(f"Recording saved at {OUTPUT_PATH}")
