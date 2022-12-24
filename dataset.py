import glob
import json
import os
import xmltodict

from typing import Any, Tuple, Optional, Callable, List, Dict, Union

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import preprocessing
from torch import Tensor
from torch.utils.data import DataLoader

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg


def encode_class(label_map, vector, num_classes) -> Tensor:

    labels_class = [label_map[label] for label in vector]
    labels_class = torch.tensor(labels_class)
    one_hot_labels = F.one_hot(labels_class, num_classes)
    return one_hot_labels


class CsPlayer(VisionDataset):
    BASE_FOLDER = "cs_agent_dataset"

    def __init__(
            self,
            root: str,
            split: str,
            transform: Optional[Callable] = None,

    ) -> None:
        super().__init__(root=os.path.join(root, self.BASE_FOLDER), transform=transform)
        self.split = verify_str_arg(split, "split", ("train", "val", "test"))
        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []
        if self.split in ("train", "test", "val"):
            self.parse_files()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        print(self.img_info[index]["img_path"])
        img = Image.open(self.img_info[index]["img_path"])

        target = self.img_info[index]["annotations"]
        side =self.img_info[index]["annotations"]['side']


        if self.transform is not None:
            img, target = self.transform(img, target)


        return img, target

    def __len__(self) -> int:
        return len(self.img_info)

    def parse_files(self):
        if self.split == "train":
            self.root = os.path.join(self.root, "train")
        if self.split == "val":
            self.root = os.path.join(self.root, "val")
        if self.split == "test":
            self.root = os.path.join(self.root, "test")

        for filename in os.listdir(self.root):
            if not filename.endswith('.xml'): continue
            fullname = os.path.join(self.root, filename)
            with open(fullname, "r") as f:
                xml_string = f.read()
            xml_dict = xmltodict.parse(xml_string)
            labels_tensor = None
            if 'object' in xml_dict['annotation']:
                labels = []
                labels_class = []
                labels_len = len(xml_dict['annotation']['object'])
                if labels_len < 6:
                    for label in xml_dict['annotation']['object']:
                        labels.append(label['bndbox'])
                        labels_class.append(label['name'])


                    items_labels = []

                    for label in labels:
                        items = []
                        for _, item in label.items():
                            items.append(float(item))
                        items_labels.append(items)
                        labels_tensor = torch.tensor(items_labels)
                    one_hot_labels = encode_class(label_map, labels_class, 2)

                    img_path = os.path.join(self.root, xml_dict['annotation']['filename'])
                    self.img_info.append(
                            {
                                "img_path": img_path,
                                "annotations": {
                                    "bbox": labels_tensor[:, 0:4],  # x, y, width, height
                                    "side": one_hot_labels[:,]

                                },
                            }
                        )




                if labels_len == 6:
                    labels = xml_dict['annotation']['object']['bndbox']
                    labels_class = xml_dict['annotation']['object']['name']
                    items = []
                    for _, item in labels.items():
                        items.append(float(item))
                    labels_tensor = torch.tensor(items)

                    label_map = {'T': 0, 'CT': 1,'C': 1}
                    one_hot_labels=encode_class(label_map,labels_class,3)

                    img_path = os.path.join(self.root, xml_dict['annotation']['filename'])
                    self.img_info.append(
                        {
                            "img_path": img_path,
                            "annotations": {
                                "bbox": labels_tensor[0:4],  # x, y, width, height
                                "side": one_hot_labels

                            },
                        }
                    )
