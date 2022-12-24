
from torch.utils.data import DataLoader

from torchvision.transforms import  PILToTensor, Resize

from dataset import CsPlayer
from torchvision import transforms

size =(320,320)
transform= transforms.Compose([PILToTensor()])
batch_size = 16
dataset = CsPlayer("D:\datasety\cs_agent_dataset\\train\_annotations.coco.json","D:\datasety\cs_agent_dataset",transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size)


for image,target,label in data_loader:
    print(label)

