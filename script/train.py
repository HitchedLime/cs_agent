
from torch.utils.data import DataLoader

from torchvision.transforms import  PILToTensor, Resize

from dataset import CsPlayer
from transforms.transforms_custom import ResizeBoundingBoxes,Compose

size =(320,320)
trasforms = Compose([ResizeBoundingBoxes(size=(520, 520)),])
batch_size = 8
dataset = CsPlayer(root="D:\git", split="train",transform=trasforms)
data_loader = DataLoader(dataset, batch_size=batch_size)


for data in data_loader:
    print(data)
