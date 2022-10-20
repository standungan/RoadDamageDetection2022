import random
import torchvision.transforms.functional as F
import torch
# buat fungsi transformer : 
# kegunaannya : 

    # if train
        # flip image + bbox with 50% chance 

    # resize image + bbox fractional form
    # konversi gambar ke tensor
    # normalize image
    # return
        
def transformer(image, bbox, label, split):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # if split == "TRAIN":
    #     if random.random() < 0.5:
    #         # chance for Flip
    
    # resize
    image = image
    # to tensor
    image = F.to_tensor(image)
    # normalize
    image = F.normalize(image, mean=mean, std=std)
    
    return image, bbox, label

def flipHorizontally(image, bbox):
    image2 = F.hflip(image)
    
    bbox2 = bbox
    bbox2[0] = image.width - bbox[0] - 1
    bbox2[2] = image.width - bbox[2] - 1
    
    temp = bbox[0]
    bbox2[0] = bbox2[2]
    bbox2[2] = temp
    
    return image2, bbox2

def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [F.adjust_brightness,
                   F.adjust_contrast,
                   F.adjust_saturation,
                   F.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image
    
def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = F.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


from utilities import photometric_distort, resize, flipHorizontally
import torchvision.transforms.functional as F
import random

def transform(image, bboxes, labels):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    new_image = image
    
    new_bboxes = bboxes
    
    new_labels = labels
    
    new_image = photometric_distort(new_image)
    
    if random.random() < 0.5:
        new_image, new_boxes = flipHorizontally(new_image, new_bboxes)

    new_image, new_bboxes = resize(new_image, new_bboxes, dims=(300, 300))
    
    new_image = F.to_tensor(new_image)
    
    new_image = F.normalize(new_image, mean=mean, std=std)       

    return new_image, new_bboxes, new_labels

