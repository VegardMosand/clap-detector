import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F

n_fft = 1000
hop = 100

def create_2d_numpy(filename : str) -> np.ndarray:
    waveform, sr = librosa.load(filename, sr = 16000)

    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop)
    # Convert to power spectrogram
    PS = np.abs(stft)**2
    return librosa.power_to_db(PS, ref=np.max)      

# Dont do power to DB?

def create_dataset(directory) -> Dataset:
    clap_bool_list : list = []
    
    soundlist : list = []    
     
    print("claps")    
    clapdir = directory + "/claps"
    for filename in os.listdir(clapdir):
        f = os.path.join(clapdir, filename)
        new = create_2d_numpy(f)
        soundlist.append(new)
        clap_bool_list.append(1.0)

    nonclapdir = directory + "/not-claps"
    print("non_claps")    
    for filename in os.listdir(nonclapdir):
        f = os.path.join(nonclapdir, filename)
        new = create_2d_numpy(f)
        soundlist.append(new)
        clap_bool_list.append(0.0)

    clap_bool_np_array : np.ndarray = np.array(clap_bool_list)
    soundlist : np.ndarray = np.array(soundlist)
    dataset : CustomClapDataset = CustomClapDataset(soundlist, clap_bool_np_array)
    return dataset

def plot(data):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(data, sr=len(data), x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.tight_layout()
    plt.show()

def plot_claps():
    nd = "/home/vegard/Documents/clap_bin/saag.wav"
    plot(create_2d_numpy(nd))
    clap = "../sounds/test_data/claps/20240228-230218.wav"
    clap2 = "../sounds/test_data/claps/20240228-230242.wav"
    print("claps:")
    plot(create_2d_numpy(clap))
    plot(create_2d_numpy(clap2))
    print("non claps:")
    non_clap = "../sounds/test_data/not-claps/20240228-221520.wav"
    non_clap2 = "../sounds/test_data/not-claps/20240228-221131.wav"
    non_clap3 = "../sounds/test_data/not-claps/20240228-221426.wav"
    non_clap4 = "../sounds/test_data/not-claps/20240228-221756.wav"
    plot(create_2d_numpy(non_clap))
    plot(create_2d_numpy(non_clap2))
    plot(create_2d_numpy(non_clap3))
    plot(create_2d_numpy(non_clap4))

def main():
    
    batch_size = 1
    epochs = 1000 
    best_loss = float("inf") 
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

    loss_fn = nn.BCEWithLogitsLoss()
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    for epoch in range(epochs):
        print("Epoch ", epoch)
        train(train_dataloader, model, loss_fn, optimizer)
        new_loss = test(test_dataloader, model, loss_fn)
        if  new_loss < best_loss:
            best_loss = new_loss
            patience = 0
            # Save the model if you want to keep the best one
            torch.save(model.state_dict(), "../target/the_clapper.pth")
        else:
            patience += 1
        
        if patience == 10:
            print("Patience spent")
            break
    return

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, Y.unsqueeze(0))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, Y.unsqueeze(0)).item()
            if (test_loss > 0.8):
                correct += 1
    test_loss /= num_batches
    correct /= size
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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Assuming mono channel input
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)


        self.reduced_num_frequencies = 1025
        self.downsampled_width = 7
        self.num_classes = 1

        # Additional convolutional layers can be added here
        self.fc1 = nn.Linear(32 * 256, 512)
        self.fc2 = nn.Linear(512, self.num_classes)  # num_classes is the number of output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        final_downsampled_width = 32
        final_reduced_num_frequencies = 256
        # Additional convolutional operations can be applied here
        x = x.view(-1, final_reduced_num_frequencies * final_downsampled_width)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#    def __init__(self):
#        super(NeuralNetwork, self).__init__()
#        self.conv_stack = nn.Sequential(
#            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),  # Assuming mono-channel input
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=(2, 2)),
#            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=(2, 2)),
#        )
#        self.flatten = nn.Flatten()
#        reduced_num_frequencies = 1025
#        downsampled_width = 7
#        num_classes = 1
#        
#        self.linear_relu_stack = nn.Sequential(
#            nn.Linear(64 * reduced_num_frequencies * downsampled_width, 512),  # Adjusted for pooling
#            nn.ReLU(),
#            nn.Linear(512, num_classes)
#        )
#
#    def forward(self, x):
#        x = self.conv_stack(x)
#        x = self.flatten(x)
#        logits = self.linear_relu_stack(x)
#        return logits

main()

# https://chat.openai.com/c/5e70a21e-8dc7-45c2-af6b-8959c6bf8d59
# Ses siste svaret