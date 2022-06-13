import torch.nn.functional as F
import torch.nn as nn
import torch

"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d( 3, 300,kernel_size=1,stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d( 300, 50,kernel_size=1,stride=1, padding=0)
        self.fc1 = nn.Linear(498750, 50)
        
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)


    def forward(self, x):
       
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print("we before flatten")
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        print("shape of x after pool",x.size())
        print("we befor linear1")
        x = F.relu(self.fc1(x))
        print("we after linear1")
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        print("size of output",len(x))
        return x 
 
"""

class Net(nn.Module):
    def __init__(self, use_global_average_pooling: bool = False):
        super().__init__()
        self.use_global_average_pooling = use_global_average_pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        if use_global_average_pooling:
            self.fc_gap = nn.Linear(64, 10)
        else:
            self.fc_1 = nn.Linear(54 * 54 * 64, 84)  # 54 img side times 64 out channels from conv2
            self.fc_2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # img side: (224 - 2) // 2 = 111
        x = self.pool(F.relu(self.conv2(x)))  # img side: (111 - 2) // 2 =  54
        if self.use_global_average_pooling:
            # mean for global average pooling (mean over channel dimension)
            x = x.mean(dim=(-1, -2))
            x = F.relu(self.fc_gap(x))
        else:  # use all features
            x = torch.flatten(x, 1)
            x = F.relu(self.fc_1(x))
            x = self.fc_2(x)
        return x



