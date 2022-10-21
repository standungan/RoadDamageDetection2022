import torch
import os
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset
from utilities import transform
labels_idx = ["D00","D10","D20","D40"]

class RoadDamageDataset(Dataset):
    def __init__(self, folder):
        
        self.dir_xmls = os.path.join(folder, "annotations", "xmls")
        self.dir_imgs = os.path.join(folder, "images")
        
        self.annotations = [os.path.join(self.dir_xmls, xml) for xml in os.listdir(self.dir_xmls)]
        self.images_file = [os.path.join(self.dir_imgs, img) for img in os.listdir(self.dir_imgs)]

    def __getitem__(self,i):
        
        image = Image.open(self.images_file[i])
        tree = ET.parse(self.annotations[i])
        objects = tree.findall("object")

        bboxes = []
        labels = []
        
        for i, obj in enumerate(objects):
            objectID = "object_{:02d}".format(i)
            name = obj.find("name").text
            if name == 'Repair':
                continue
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            label = labels_idx.index(name)
            labels.append(label)
            bboxes.append([xmin, ymin, xmax, ymax])
        
        bboxes = torch.FloatTensor(bboxes)
        labels = torch.FloatTensor(labels)
        
        image, bboxes, labels = transform(image, bboxes, labels)
        
        return image, bboxes, labels

    def __len__(self):
        
        return len(self.images_file)