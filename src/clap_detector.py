import time
import pathlib  
import threading
import os
import librosa
import torch
import numpy as np
from torch import nn
from clap_nn import ClapDetectNN
from clap_utils import create_PS

sample_rate = 16000

filename = ""
lock = threading.Lock()

def recorder():
    lock.acquire()
    while True:
        filename = time.time()
        pathlib.Path(filename).touch()
        lock.release()
        # Record 48 hours of audio
        f"arecord -d 172800 -D plughw:0 -c1 -r {sample_rate} -f S32_LE -t wav -V mono {filename}"
        lock.acquire()
        pathlib.Path.unlink(filename)

def clap_listener():
    model = ClapDetectNN()
    model.load_state_dict(torch.load("../target/the_clapper.pth"))
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    lock.acquire()
    working_file = filename
    stream = librosa.stream(filename, 1, 3200, 1600, False)
    
    for block in stream:
        lock.release()
        power_spectrogram = create_PS(block)
        prediction = model(power_spectrogram)
        loss : torch.tensor = loss_fn(prediction, torch.tensor(1.0))
        if (loss < 0.5):
            print("Detected clap!!")
        lock.acquire()
        if not os.path.isFile(working_file):
            stream = librosa.stream(filename, 1, 3200, 1600, False)
            working_file = filename
