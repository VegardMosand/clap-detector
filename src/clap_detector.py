import time
import pathlib  
import threading
import os

sample_rate = 16000

filename = ""
lock = threading.Lock()

def recorder():
    while True:
        filename = time.time()
        pathlib.Path(filename).touch()
        lock.release()
        # Record 48 hours of audio
        f"arecord -d 172800 -D plughw:0 -c1 -r {sample_rate} -f S32_LE -t wav -V mono {filename}"
        lock.acquire()
        pathlib.Path.unlink(filename)

def clap_listener():
    while os.path.isfile(filename):
        
