
import numpy as np
import matplotlib.pyplot as plt
import torch
import dataset 
import os 
from torch.utils.data import  DataLoader
import torch.nn as nn

import torchvision
import check_device

import neural_network
import torch.optim as optim

EPS = 1.e-7
LR=0.5
WEIGHT_DECAY=0.5
batch_size =50
#DATA LOADING ###################################################################################################################
#transforms= 
    #torchvision.transforms.ToTensor(),ms.Comp
#    
 #   torchvision.transforms.Resize((535, 535))],
#)

transforms = torch.nn.Sequential(
    torchvision.transforms.CenterCrop(10),
    torchvision.transforms.Resize((300, 535)),
)

test_dataset =dataset.csHeadBody(csv_file="cs_agent\\images\\test_labels.csv",root_dir="cs_agent\\images\\test",transform=torchvision.transforms.Compose([transforms,torchvision.transforms.ToTensor()]))
train_dataset =dataset.csHeadBody(csv_file="cs_agent\\images\\train_labels.csv",root_dir="cs_agent\\images\\train",transform=transforms)
train_loader =DataLoader(dataset =train_dataset,batch_size=batch_size,shuffle=True)
test_loader =DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)




#DATA LOADING ###################################################################################################################END


#NEURAL NET #####################################################################################################################################################

net=neural_network.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#NEURAL NET END ######################################################################################



for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')