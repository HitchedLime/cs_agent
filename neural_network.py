import torch.nn.functional as F
import torch.nn as nn
import torch


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
 



