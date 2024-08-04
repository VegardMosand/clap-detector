from torch import nn
import torch.nn.functional as F

# Define model
class ClapDetectNN(nn.Module):
    def __init__(self):
        super(ClapDetectNN, self).__init__()
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
        output = output.view(-1, 47520)
        output = self.fc1(output)
        return output 