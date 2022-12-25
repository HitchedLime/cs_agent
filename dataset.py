import json
import os
import xmlrpc.client

import cv2
import numpy as np
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image


class CsPlayer(Dataset):
    def __init__(self, json_file, root_dir, split="train",transforms=None):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)

        self.transforms = transforms
        self.split = split
        self.img_info = []
        self.parse_json()
        self.root_dir = os.path.join(root_dir, split)
    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        img_info = self.img_info[idx]

        image_path = os.path.join(self.root_dir, img_info['file_name'])

        image = Image.open(image_path)

        if self.transforms is not None:
            image = self.transforms(image)

        target ={}
        target['bbox']=img_info['bbox']
        target["category"]=img_info["category"]




        return image,target
    def parse_json(self):
        if "images" in self.coco:


            for img in self.coco['images']:
                for box in self.coco['annotations']:
                    if img['id']==box['image_id']:
                        label_map = {1: 0, 2: 1}
                        category_labels = encode_class(label_map,[box['category_id']],2)
                        box_tensor = torch.Tensor(box['bbox'])
                        self.img_info.append({ "file_name":img['file_name'],
                                               "bbox":  box_tensor,
                                               "category":category_labels


                        })



def encode_class(label_map, vector, num_classes):

    labels_class = [label_map[label] for label in vector]
    labels_class = torch.tensor(labels_class)
    one_hot_labels = F.one_hot(labels_class, num_classes)
    return one_hot_labels