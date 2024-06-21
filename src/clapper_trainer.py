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

n_fft = 128
hop = 8
debug = False

def create_2d_numpy(filename : str) -> np.ndarray:
    waveform, sr = librosa.load(filename, sr = 16000)

    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop)
    #mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    # Convert to amp spectrogram (uncomment below for power spectrogram)
    PS = np.abs(stft) **2
    return librosa.power_to_db(PS, ref=np.max)      

# Dont do power to DB?

def create_dataset(directory) -> Dataset:
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
        librosa.display.specshow(new, sr=16000, x_axis='time', y_axis='mel')
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
        # Compute prediction and loss
        #pred = model(X)
        #loss = loss_fn(pred, Y.unsqueeze(0))

        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using sounds from the training set
        prediction = model(sound)
        # compute the loss based on model output and real labels
       # print(prediction, label.unsqueeze(0))
        loss : torch.tensor = loss_fn(prediction, label.unsqueeze(0))
        # backpropagate the loss
        loss.backward()
        # adjust parameters based on the calculated gradients
        optimizer.step()

        # Backpropagation
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        running_loss += loss.item()
        if batch % 200 == 199:
            current = batch * len(sound)
            print(f"loss: {running_loss/200:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0

def test(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    #model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in dataloader:
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
        self.dataset : torch.tensor = torch.from_numpy(dataset).unsqueeze(1) # Remove unsqueeze?
        self.labels : torch.tensor = torch.from_numpy(labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        power_spectogram = self.dataset[idx]
        label = self.labels[idx]
        return power_spectogram, label

# andrewNG concolutional neural network 
# MAX
# residual layers
# Amplitude spectogram istedenfor power spectrogram?
# Mel-frequency cepstral coefficientse


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

       # self.final_width = 32
       # self.final_reduced_num_frequencies = 256
       # self.num_classes = 1

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(3, 2) #(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Assuming mono channel input
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(47520, 1)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = self.pool(output)
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)
        output = F.relu(self.bn3(self.conv3(output)))     
        output = F.relu(self.bn4(self.conv4(output)))     
        output = output.view(-1, 47520)        # 64
        output = self.fc1(output)
        #output = torch.nn.Sigmoid(output)

        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #final_downsampled_width = 32
        #final_reduced_num_frequencies = 256
        ## Additional convolutional operations can be applied here
        #x = x.view(-1, final_reduced_num_frequencies * final_downsampled_width)  # Flatten the output for the fully connected layer
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        return output 

def main():
    batch_size = 1
    epochs = 1000 
    best_accuracy = float("inf") 
    patience = 0

    train_dataset = create_dataset("../sounds/training_data")
    test_dataset = create_dataset("../sounds/test_data")

    #plot_claps()

    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)
        
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
   # loss_fn = nn.CrossEntropyLoss()
   # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    for epoch in range(epochs):
        print("Epoch ", epoch)
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy = test(test_dataloader, model, loss_fn)
        if  accuracy < best_accuracy:
            best_accuracy = accuracy
            patience = 0
            # Save the model if you want to keep the best one
            torch.save(model.state_dict(), "../target/the_clapper.pth")
        else:
            patience += 1
        
        if patience == 10:
            print("Patience spent")
            break
    return
    
main()

# https://chat.openai.com/c/5e70a21e-8dc7-45c2-af6b-8959c6bf8d59
# Ses siste svaret