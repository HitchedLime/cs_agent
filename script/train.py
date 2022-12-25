import torch
import torchvision
import torchvision as torchvision
from PIL import ImageDraw
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.transforms import PILToTensor, Resize, ToTensor

from dataset import CsPlayer
from torchvision import transforms

size = (320, 320)
transform = transforms.Compose([ToTensor()])
batch_size = 4
dataset = CsPlayer("D:\datasety\cs_agent_dataset\\train\_annotations.coco.json", "D:\datasety\cs_agent_dataset",
                   transforms=transform)
data_loader = DataLoader(dataset, batch_size=batch_size)



def compute_loss(output, target, labels):
    classification_output, output_bb = output
    bb_loss = torch.nn.SmoothL1Loss(output_bb, target)
    class_loss = torch.nn.CrossEntropyLoss(classification_output, labels)

    return bb_loss, class_loss


"""



##################################



# put the pieces together inside a FasterRCNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
images,targets,labels = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)
"""

############################


anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)
#AttributeError: module 'torchvision.models.detection' has no attribute 'ssd_resnet50_fpn
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 10
model.eval()



for i in range(epochs):
    for data in data_loader:
        image, target = data

        optimizer.zero_grad()
        with torch.no_grad():
            output = model(image)


    # loss = compute_loss(output,target,label)
    # loss.backward()
    #  optimizer.step()

    # Update model weights
