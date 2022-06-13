import os
import pandas as pd
from torchvision.io import read_image
import re
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
import torch

class csHeadBody(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        print("thos os label",label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
     #head is one body is 2
        

        """ class csHeadBody ( Dataset ) :
        def __init__( self , csv_file , root_dir , transform = None ) :
            self . annotations = pd . read_csv ( csv_file )
            self.root_dir = root_dir
            self.transform = transform
        def __len__( self ) :
            return len ( self . annotations ) # 25000
        def __getitem__ ( self , index ) :
            img_path = os.path.join ( self.root_dir , self.annotations.iloc [ index , 0 ] )
            image = io.imread ( img_path )
            y_label = torch . tensor ( int ( self . annotations.iloc [ index , 1 ] ) )
            if self.transform :
                image = self.transform ( image )
            return ( image , y_label ) """
