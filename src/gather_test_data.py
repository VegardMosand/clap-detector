import subprocess
import datetime
import os
import soundfile as sf
from typing import NoReturn, Optional

# Ensure the directories exist
os.makedirs('claps', exist_ok=True)
os.makedirs('not-claps', exist_ok=True)

# Used for deletion
last_recorded_file: dict[str, Optional[str]] = {'claps': None, 'not-claps': None}
sample_rate = 16000
samples = 3200
file_count = 0
non_claps = 2000

def record_clap():
    global last_recorded_file
    global file_count
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename: str = f"claps/{timestamp}.wav"
    print("Recording 60s of claps!")
    command: str = f"arecord -d 240 -D plughw:0 -c1 -r {sample_rate} -f S32_LE -t wav -V mono {filename}" # 240 seconds recording
    subprocess.call(command, shell=True)
    file_count += 1
    print(f"Recorded in claps mode. Total files: {file_count}")
    last_recorded_file = filename  # Update the last recorded file for the current mode
    
def record_non_claps():
    global file_count
    print(f"Recording {non_claps} non-claps!")
    for _ in range(0, non_claps):
        timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename: str = f"not-claps/{timestamp}.wav"
        command: str = f"arecord -s 22000 -D plughw:0 -c1 -r 16000 -f S32_LE -t wav -V mono {filename}"
        subprocess.call(command, shell=True)
        # audio has to be cut to size. The reason for the long recording is that the start and end of each recording is tainted
        file, sr = sf.read(filename, start=samples*3, stop=samples*4)
        sf.write(filename, file, sr, subtype='PCM_16')
        file_count += 1
        print(f"Recorded in not-claps mode. Total files: {file_count}")

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
print("Press Enter to record, Space to record non-claps, 'd' to delete last file, 'q' to quit.")
print(f"Starting in {mode} mode.")

while True:
    user_input: str = input()
    
    if user_input == 'q':
        break
    elif user_input == ' ':
        record_non_claps()
        break
    elif user_input == 'd':
        remove_last_recorded_file()
    else:
        record_clap() 