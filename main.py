
import numpy as np
import matplotlib.pyplot as plt
import torch
import dataset 
import os 
from torch.utils.data import  DataLoader
import torch.nn as nn
import pandas as pd 
import torchvision
import check_device
from sklearn import preprocessing
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: x / 255.),
    torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    torchvision.transforms.Resize((224, 224)),
])

test_dataset =dataset.csHeadBody(csv_file="cs_agent\\images\\test_labels.csv",root_dir="cs_agent\\images\\test",transform=transforms)
train_dataset =dataset.csHeadBody(csv_file="cs_agent\\images\\train_labels.csv",root_dir="cs_agent\\images\\train",transform=transforms)
train_loader =DataLoader(dataset =train_dataset,batch_size=batch_size,shuffle=True)
test_loader =DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)



#DATA LOADING ###################################################################################################################END


#NEURAL NET #####################################################################################################################################################

net=neural_network.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
le = preprocessing.LabelEncoder()

#NEURAL NET END ######################################################################################



for epoch in range(2):  # loop over the dataset multiple times
   # print("train_loader",train_loader.__len__())
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #print(data)
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        
       # print("theese are ouptus ",outputs)
        loss = criterion(outputs,torch.as_tensor(le.fit_transform(labels)))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
          # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
        running_loss = 0.0

print('Finished Training')
PATH = './test_net.pth'
torch.save(net.state_dict(), PATH)
print("Model saved !")


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(test_loader)
images, labels = dataiter.next()
classes= (1,2)
# print images
#imshow(torchvision.utils.make_grid(images))


net = neural_network.Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')