import numpy as np
import librosa

n_fft = 128
hop = 8

def create_2d_numpy(filename : str) -> np.ndarray:
    block, sr = librosa.load(filename, sr = 16000)
    return create_PS(block)

def create_PS(block):
    stft = librosa.stft(block, n_fft=n_fft, hop_length=hop)
    PS = np.abs(stft) **2
    return librosa.power_to_db(PS, ref=np.max)      
    