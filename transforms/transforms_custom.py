import cv2
import torch
from PIL.Image import Image
from torchvision import transforms


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        image_to_tensor = transforms.PILToTensor()
        for t in self.transforms:
            image, target = t(image, target)

        print(type(image), type(target))
        image = image_to_tensor(image)

        return image, target


class ResizeBoundingBoxes(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, boxes):
        # resize the image

        # plot_image_with_boxes(image, boxes["bbox"])
        image = image.resize(self.size)
        width, height = image.size
        # resize the bounding boxes

        boxes_ = boxes["bbox"]

        if torch.is_tensor(boxes_) and len(boxes_)==4:
            boxes_= boxes_.numpy()

            boxes_[0] = boxes_[0] * self.size[0] / width
            boxes_[1] = boxes_[1] * self.size[1] / height
            boxes_[2] = boxes_[2] * self.size[0] / width
            boxes_[3] = boxes_[3] * self.size[1] / height

            return image, boxes_
        else:


            after_transform = []
            boxes_=boxes_.numpy()
            for box in boxes_:
                    box[0] = box[0].item() * self.size[0] / width
                    box[1] = box[1].item() * self.size[1] / height
                    box[2] = box[2].item() * self.size[0] / width
                    box[3] = box[3].item() * self.size[1] / height

                    after_transform.append(box)


        # plot_image_with_boxes(image, boxes["bbox"])

        return image, after_transform


from PIL import ImageDraw


def plot_image_with_boxes(image, boxes):
    # create a copy of the image
    image = image.copy()
    draw = ImageDraw.Draw(image)
    if torch.is_tensor(boxes):
        xmin, xmax, ymin, ymax = boxes[0], boxes[1], boxes[2], boxes[3]
        draw.rectangle((xmin, ymin, xmax, ymax), outline='red')

    else:
        for box in boxes:
            xmin, xmax, ymin, ymax = box[0], box[1], box[2], box[3]
            draw.rectangle((xmin, ymin, xmax, ymax), outline='red')

    # labels_tensor = torch.tensor([labels[0]['coordinates']['x'], labels[0]['coordinates']['y'],
    #  labels[0]['coordinates']['width'], labels[0]['coordinates']['height']])
    # display the image
    image.show()
