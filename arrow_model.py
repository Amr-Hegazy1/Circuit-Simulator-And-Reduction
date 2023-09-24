import torch.nn as nn
import torch.nn.functional as F
import torch

class ArrowModel(nn.Module):
    def __init__(self,num_classes=4):
        super().__init__()
        
        # image is 128x128x3
        # after 1st conv: (128-5+1)/1 = 124x124x6
        # after 1st pooling: 62x62x6
        self.conv1 = nn.Conv2d(3, 6, 5)
        # after 2nd conv: (62-5+1)/1 = 58x58x16
        # after 2nd pooling: 29x29x16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # after 1st fc: 16*29*29 = 13456
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        
        
        
        
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



