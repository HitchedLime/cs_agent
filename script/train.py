
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, PILToTensor, Resize

from dataset import CsPlayer
size =(320,320)
trasforms = Compose([PILToTensor(),Resize(size)])
batch_size = 8
dataset = CsPlayer(root="D:\git", split="train",transform=trasforms)
data_loader = DataLoader(dataset, batch_size=batch_size)


for data in data_loader:
    print(data)
