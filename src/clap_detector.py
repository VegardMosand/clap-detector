import time
import pathlib  
import threading
import os
import alsaaudio
import sys
print("Importing torch. This may take a while...")
import torch
import numpy as np
import librosa
#import soundfile as sf
import subprocess
from torch import nn
from clap_nn import ClapDetectNN
from clap_utils import create_PS 
from clap_utils import create_2d_numpy
import wave

sample_rate = 16000
sensitivity = 0.55 #0.05

livefile = "wearedoingitlive.wav"
filename = "2claps_with_noise.wav"
filename1 = "/home/vegard/Documents/sshfs/clapper/src/122.wav"

def record():
    print("STARTING RECORDING")
    # Record 48 hours of audio
    subprocess.run(f"arecord -d 172800 --quiet -D plughw:1 -c 1 -r {sample_rate} -f S32_LE -t wav -V mono {livefile} --quiet", shell=True)

# generates chunks of the correct size and format
def chunk_generator():
    # Hop length of 100 ms. Each segment/chunk is 200 ms
    stream = librosa.stream(path=sys.argv[1], block_length=1, frame_length=3200, hop_length=1600, mono=True, dtype=np.float32)
    build_chunk : np.ndarray = np.array([])
    for chunk in stream:
        # Handle bad chunk size
        if (chunk.size != 3200):
            remaining_chunk_size = 3200 - build_chunk.size
            if (remaining_chunk_size <= chunk.size):
                build_chunk = np.append(build_chunk, chunk[:remaining_chunk_size])
                yield build_chunk
                build_chunk = chunk[remaining_chunk_size:]
                continue
                # complete, ready to yield
                # yield then set rest of chunk to build_chunk
            else:
                build_chunk = np.append(build_chunk, chunk)
                continue
        yield chunk

def chunk_generator2():
    new_chunk : np.ndarray
    build_chunk : np.ndarray
    # Each period will be half a chunk. This is to be able to make the overlap
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device='plughw:1,0', rate=sample_rate, periodsize=1600, channels=1, format=alsaaudio.PCM_FORMAT_FLOAT_LE)
    length, data = inp.read()
    build_chunk = np.frombuffer(data, dtype=np.float32)
    while True:
        length, data = inp.read()

        if length > 0:
            new_chunk = np.frombuffer(data, dtype=np.float32)

            yield np.append(build_chunk, new_chunk)
            build_chunk = new_chunk

def clap_listener():
    model = ClapDetectNN()
    model.load_state_dict(torch.load("../target/the_clapper.pth"))
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    # hop length is half of frame length to have 50% overlap. This is to make sure a clap is at the 
    # start (at least almost) of a frame. To avoid a clap being counted twice the first chunk after a clap detection is ignored

    #stream = librosa.stream(path=livefile, block_length=1, frame_length=3200, hop_length=1600, mono=True, dtype=np.float32)
    #losslist = []
    stream =  chunk_generator2()
    reslist = [1.0]
    reslist = np.array(reslist)
    reslist : torch.tensor = torch.from_numpy(reslist)
    claps = 0
    i = 0

    prevclap = False

    with torch.no_grad():
        for chunk in stream:
            if(prevclap):
                prevclap =False
                continue

            if (i % 1000 == 0):
                print(f"Got frame number {i}")

            i += 1
            soundlist = []

            power_spectrogram = create_PS(chunk)
            #power_spectrogram = create_2d_numpy(filename1)

            soundlist.append(power_spectrogram)
            soundlist = np.array(soundlist)

            input_tensor = torch.from_numpy(soundlist).unsqueeze(1)
            prediction = model(input_tensor)

            # Calculate loss
            loss : torch.tensor = loss_fn(prediction, reslist.unsqueeze(0))

            if (loss < sensitivity):
                claps += 1
                #losslist.append(loss)
                print(f"Detected clap at {i*100}ms!! This was clap number {claps}. Loss: {loss}")
                #sf.write(f"../candidates/candidate_{i}.wav", chunk, 16000)
                prevclap = True
            else:
                prevclap = False
                #print(f"not a clap!! Loss {loss}")
    
    print(f"Done. Total number of claps: {claps}.") #losslist:\n{losslist}")


#t1 = threading.Thread(target=record, args=[])
#t1.start()
clap_listener()
#t2 = threading.Thread(target=clap_listener, args=[])

#t1.start()
#time.sleep(0.1)
#t2.start()