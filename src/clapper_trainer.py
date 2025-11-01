import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F
from torch.optim import Adam
from audio_visualizer import *
from clap_utils import create_2d_numpy

from clap_nn import ClapDetectNN

debug = False

def create_dataset(directory) -> Dataset:
    # 1.0 if clap. 0.0 if not
    clap_bool_list : list = []
    
    soundlist : list = []    
    i = 0 
    print("claps")    
    clapdir = directory + "/claps"
    for filename in os.listdir(clapdir):
        i += 1
        f = os.path.join(clapdir, filename)
        new = create_2d_numpy(f)
        soundlist.append(new)
        clap_bool_list.append(1.0)
        if (debug and i % 100 == 0):
            display_power_spectrogram(new)
            

    nonclapdir = directory + "/not-claps"
    print("non_claps")    
    for filename in os.listdir(nonclapdir):
        i += 1
        f = os.path.join(nonclapdir, filename)
        new = create_2d_numpy(f)
        soundlist.append(new)
        clap_bool_list.append(0.0)
        if (debug and i % 100 == 0):
            display_power_spectrogram(new)

    clap_bool_np_array : np.ndarray = np.array(clap_bool_list)
    soundlist : np.ndarray = np.array(soundlist)
    dataset : CustomClapDataset = CustomClapDataset(soundlist, clap_bool_np_array)
    return dataset

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #model.train()
    running_loss = 0
    for batch, (sound, label) in enumerate(dataloader):
        optimizer.zero_grad()
        prediction = model(sound)
        loss : torch.tensor = loss_fn(prediction, label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 200 == 199:
            current = batch * len(sound)
            print(f"loss: {running_loss/200:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0

def test(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in dataloader:
            print(X.shape)
            pred = model(X)
            loss = loss_fn(pred, Y.unsqueeze(0)).item()
            test_loss += loss
            if (loss < 0.5):
                correct += 1
    test_loss = test_loss / num_batches
    correct = correct / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.7f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


class CustomClapDataset(Dataset):
    def __init__(self, dataset, labels, transform=None, target_transform=None):
        self.dataset : torch.tensor = torch.from_numpy(dataset).unsqueeze(1)
        self.labels : torch.tensor = torch.from_numpy(labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        power_spectogram = self.dataset[idx]
        label = self.labels[idx]
        return power_spectogram, label

def main():
    batch_size = 1
    epochs = 1000 
    best_accuracy = float("inf") 
    patience = 0

    train_dataset = create_dataset("../sounds/training_data")
    test_dataset = create_dataset("../sounds/test_data")

    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)
        
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = ClapDetectNN().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    for epoch in range(epochs):
        print("Epoch ", epoch)
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy = test(test_dataloader, model, loss_fn)
        if  accuracy < best_accuracy:
            best_accuracy = accuracy
            patience = 0
            #torch.save(model.state_dict(), "../target/the_clapper.pth")
        else:
            patience += 1
        
        if patience == 10:
            print("Patience spent")
            break
    return
    
main()