import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def display_power_spectrogram(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Compute the Short-Time Fourier Transform (STFT) of the audio
    D = librosa.stft(y)

    # Convert the amplitude spectrogram to a power spectrogram (magnitude squared)
    S = np.abs(D)**2

    # Convert to decibels for better visualization
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot the power spectrogram
    plt.figure(figsize=(10, 6))
    # Display the spectrogram
    plt.imshow(S_dB, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Power/Frequency')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Power Spectrogram')
    plt.show()

audio_path = '/home/vegard/Documents/clap_dataset/1.wav'
display_power_spectrogram(audio_path)