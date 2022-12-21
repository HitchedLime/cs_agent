import glob
import os
import json
from typing import Any, Tuple, Optional, Callable, List, Dict, Union

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import DataLoader

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg


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
        img = Image.open(self.img_info[index]["img_path"])

        target = self.img_info[index]["annotations"]


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

        annotation_file = open(os.path.join(self.root,"_annotations.createml.json"))
        annotation =json.load(annotation_file)


        for i in annotation:
            labels = i['annotations']
            label_map = {'T': 0, 'CT': 1}
            labels_class = [label_map[label] for label in labels[0]['label']]
            labels_class = torch.tensor(labels_class)
            one_hot_labels = F.one_hot(labels_class,2)

            labels_tensor = torch.tensor([labels[0]['coordinates']['x'],labels[0]['coordinates']['y'],
                                         labels[0]['coordinates']['width'],labels[0]['coordinates']['height']])

            img_path = os.path.join(self.root, i["image"])
            self.img_info.append(
                {
                    "img_path": img_path,
                    "annotations": {
                        "bbox": labels_tensor[0:4] , # x, y, width, height
                        "side": one_hot_labels


                    },
                }
            )










