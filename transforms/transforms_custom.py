import cv2
import torch
from PIL.Image import Image
from torchvision import transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image,target)

        return image, target




class ResizeBoundingBoxes(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, boxes):
        # resize the image
        image =  image.resize(self.size)

        # resize the bounding boxes


        width,height = image.size
        boxes_=boxes["bbox"]
        after_transform=[]
        for box in boxes_:
            box[0] = box[0] * self.size[0] / width
            box[1] = box[1] * self.size[1] / height
            box[2] = box[2] * self.size[0] / width
            box[3] = box[3] * self.size[1] / height
            after_transform.append(box)
        boxes["bbox"]=after_transform
        plot_image_with_boxes(image, boxes)



        return image, boxes

from PIL import  ImageDraw

def plot_image_with_boxes(image, boxes):
    # create a copy of the image
    image = image.copy()
    draw = ImageDraw.Draw(image)

    # draw the bounding boxes on the image
    for box in boxes["bbox"]:
        xmin, ymin, xmax, ymax = box
        draw.rectangle((xmin, ymin, xmax, ymax), outline='red')

    # display the image
    image.show()
