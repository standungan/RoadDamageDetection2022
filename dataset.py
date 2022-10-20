import torch
import os
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from utilities import transformer

cat_name_to_id = {
    "D00" : 1,
    "D10" : 2,
    "D20" : 3,
    "D40" : 4,
    "Repair" : 5,
    }

class RoadDamageDataset(Dataset):
    
    def __init__(self, dataset_name,split="TRAIN"):
        self.dataset = pd.read_csv(dataset_name)    
        self.filename = self.dataset.filename
        
        self.xmin = self.dataset.xmin
        self.ymin = self.dataset.ymin
        self.xmax = self.dataset.xmax
        self.ymax = self.dataset.ymax
        
        self.labels = self.dataset.name
        self.split = split
        
    def __getitem__(self, i):
        img = Image.open(self.filename[i])
        img = img.convert('RGB')
        
        bbox = torch.as_tensor([self.xmin[i], self.ymin[i], self.xmax[i], self.ymax[i]], 
                               dtype=torch.float32)
        
        label = torch.as_tensor(cat_name_to_id[self.labels[i]], 
                                dtype=torch.float32)
        
        img, bbox, label = transformer(img, bbox, label, self.split)
        
        return img, bbox, label
    
    def __len__(self):
        return len(self.dataset)
    
    