import numpy as np
import torch
import dataset 
from torch.utils.data import  DataLoader
import torch.nn as nn
import torchvision
from sklearn import preprocessing
import neural_network
import torch.optim as optim
import os

EPS = 1.e-7
LR=0.5
WEIGHT_DECAY=0.5
PATH = './test_net.pth'
ROOT_DIR = 'images\\'


train_path = 'images\\test_labels.csv'
test_path ='images/test_labels.csv'



#DATA LOADING ###################################################################################################################
batch_size =5
number_of_epochs=50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: x / 255.),
    torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    torchvision.transforms.Resize((224, 224)),
])

test_dataset =dataset.csHeadBody(csv_file=test_path,root_dir=ROOT_DIR+'test',transform=transforms)
train_dataset =dataset.csHeadBody(csv_file=train_path,root_dir=ROOT_DIR+'train',transform=transforms)
train_loader =DataLoader(dataset =train_dataset,batch_size=batch_size,shuffle=False)
test_loader =DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


#DATA LOADING ###################################################################################################################END


#NEURAL NET #####################################################################################################################################################

net=neural_network.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
le = preprocessing.LabelEncoder()

#NEURAL NET END ######################################################################################



for epoch in range(number_of_epochs):  # loop over the dataset multiple times
   
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
       
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
          
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
        running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), PATH)
print("Model saved !")




dataiter = iter(test_loader)

images, labels = dataiter.next()
classes= (1,2)



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

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')