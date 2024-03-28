import subprocess
import datetime
import librosa
import os
import soundfile as sf
from typing import NoReturn, Optional

# Ensure the directories exist
os.makedirs('claps', exist_ok=True)
os.makedirs('not-claps', exist_ok=True)

# Track the last recorded file in each mode
last_recorded_file: dict[str, Optional[str]] = {'claps': None, 'not-claps': None}
sample_rate = 16000

def record_clap(folder: str):
    file_count: int = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename: str = f"{folder}/{timestamp}.wav"
    print("Recording 60s of claps!")
    command: str = f"arecord -d 240 -D plughw:0 -c1 -r {sample_rate} -f S32_LE -t wav -V mono {filename}" # 240 seconds recording
    # 3200
    librosa.load(filename)
    subprocess.call(command, shell=True)
    file_count += 1
    print(f"Recorded in {folder} mode. Total files: {file_count}")
    last_recorded_file[folder] = filename  # Update the last recorded file for the current mode
    
def record_non_claps(folder : str):
    print("Recording 1000 non-claps!")
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename: str = f"non-claps/{timestamp}.wav"
    samples = 3200
    command: str = f"arecord -s {samples*3} -D plughw:0 -c1 -r 16000 -f S32_LE -t wav -V mono {filename}" # Recording 1 second of non claps
    for _ in range(0, 1000):
        subprocess.call(command, shell=True)
        file = librosa.load(filename)
        cut_audio = file[samples, samples*2]
        sf.write(filename, cut_audio, sample_rate)
        file_count += 1
        print(f"Recorded in {folder} mode. Total files: {file_count}")

def remove_last_recorded_file(folder: str) -> NoReturn:
    global last_recorded_file
    filename = last_recorded_file[folder]
    if filename and os.path.exists(filename):
        os.remove(filename)
        print(f"Removed last recorded file: {filename}")
        last_recorded_file[folder] = None  # Reset the last recorded file for the current mode
    else:
        print("No file to remove.")

mode: str = 'claps'  # Start in 'claps' mode
print("Press Enter to record, Space to toggle mode, 'd' to delete last file, 'q' to quit.")
print(f"Starting in {mode} mode.")

while True:
    user_input: str = input()
    
    match user_input:
        case 'q':
            break
        case ' ':
            record_non_claps()
            break
        case 'd':
            remove_last_recorded_file()
        case _:
            record_clap() 