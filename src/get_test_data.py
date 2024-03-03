import subprocess
import datetime
import os
import time
from typing import NoReturn, Optional

# Ensure the directories exist
os.makedirs('claps', exist_ok=True)
os.makedirs('not-claps', exist_ok=True)

# Track the last recorded file in each mode
last_recorded_file: dict[str, Optional[str]] = {'claps': None, 'not-claps': None}

def record_audio(folder: str) -> NoReturn:
    global last_recorded_file
    file_count: int = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename: str = f"{folder}/{timestamp}.wav"
    time.sleep(0.5)
    print("3")
    time.sleep(0.5)
    print("2")
    time.sleep(0.5)
    print("1")
    time.sleep(0.5)
    print("0")
    time.sleep(0.1)
    print("Recording!")
    command: str = f"arecord -s 2400 -D plughw:0 -c1 -r 16000 -f S32_LE -t wav -V mono {filename}" # 0.15 seconds recording
    subprocess.call(command, shell=True)
    file_count += 1
    print(f"Recorded in {folder} mode. Total files: {file_count}")
    last_recorded_file[folder] = filename  # Update the last recorded file for the current mode

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
    
    if user_input == 'q':
        break  # Exit the loop to end the script
    elif user_input == ' ':
        mode = 'not-claps' if mode == 'claps' else 'claps'  # Toggle mode
        file_count: int = len([name for name in os.listdir(mode) if os.path.isfile(os.path.join(mode, name))])
        print(f"Mode changed to {mode}. Total files: {file_count}")
    elif user_input == 'd':
        remove_last_recorded_file(mode)
    else:
        record_audio(mode)
