import glob
import os
from typing import Any, Tuple, Optional, Callable, List, Dict, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg


class CsPlayer(VisionDataset):
    BASE_FOLDER = "cs_agent"

    def __init__(
            self,
            root: str,
            split: str,

            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

    ) -> None:
        super().__init__(root=os.path.join(root, self.BASE_FOLDER), transform=transform,
                         target_transform=target_transform)
        self.split = verify_str_arg(split, "split", ("train", "val", "test"))
        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []
        if self.split in ("train", "test", "val"):
            self.parse_files()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.img_info[index]["img_path"])

        target = self.img_info[index]["annotations"]

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)

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
        images_path = os.path.join(self.root, "images")
        labels_path = os.path.join(self.root, "labels")
        files_images = os.listdir(images_path)
        files_labels = os.listdir(labels_path)

        for i in range(len(files_images)):

            img_path = os.path.join(images_path, files_images[i])
            annotation_file = open(os.path.join(labels_path, files_labels[i]), 'r')
            labels = []
            number_boxes =0
            for line in annotation_file:
                number_boxes+=1
                labels.append(list(map(float, line.split())))
                labels_tensor = []
                for label in labels:
                    labels_tensor.append(torch.tensor(label))

                self.img_info.append(
                {
                    "img_path": img_path,
                    "annotations": {
                        "bbox": labels_tensor,  # x, y, width, height#

                    },
                }
            )
