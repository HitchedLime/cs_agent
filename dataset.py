import json
import os
import xmlrpc.client

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset

from PIL import Image


class CsPlayer(Dataset):
    def __init__(self, json_file, root_dir, split="train",transform=None):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.img_info = []
        self.parse_json()
    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img_info = self.img_info[idx]
        image_path = os.path.join(self.root_dir+"\\" +self.split, img_info['file_name'])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)




        return image,  img_info['bbox'],img_info['category']
    def parse_json(self):
        if "images" in self.coco:


            for img in self.coco['images']:
                for box in self.coco['annotations']:
                    if img['id']==box['image_id']:
                        label_map = {1: 0, 2: 1}
                        category_labels = encode_class(label_map,[box['category_id']],2)

                        self.img_info.append({ "file_name":img['file_name'],
                                               "bbox":box['bbox'],
                                               "category":category_labels


                        })



def encode_class(label_map, vector, num_classes):

    labels_class = [label_map[label] for label in vector]
    labels_class = torch.tensor(labels_class)
    one_hot_labels = F.one_hot(labels_class, num_classes)
    return one_hot_labels