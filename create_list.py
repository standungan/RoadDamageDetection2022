import pandas as pd
import os
import xml.etree.ElementTree as ET
import configs as cfg

if __name__ == '__main__':

    dir_xmls = os.path.join(cfg.train_dir, "annotations", "xmls")
    dir_imgs = os.path.join(cfg.train_dir, "images")

    annotations = [os.path.join(dir_xmls, xml) for xml in os.listdir(dir_xmls)]
    images = [os.path.join(dir_imgs, img) for img in os.listdir(dir_imgs)]

    dataset = []
    for i, annotation in enumerate(annotations):

        filename = images[i]
        tree = ET.parse(annotation)
        objects = tree.findall("object")

        for i, obj in enumerate(objects):
            objectID = "object_{:02d}".format(i)
            name = obj.find("name").text

            bbox = obj.find("bndbox")
            xmin = bbox.find("xmin").text
            ymin = bbox.find("ymin").text
            xmax = bbox.find("xmax").text
            ymax = bbox.find("ymax").text

            temp = [filename, objectID, name, xmin, ymin, xmax, ymax]
            dataset.append(temp)

    df = pd.DataFrame(dataset, columns=["filename", "objectID", "name", "xmin", "ymin", "xmax", "ymax"])

    df.to_csv(dir_imgs.split("/")[1]+".csv", index=False)