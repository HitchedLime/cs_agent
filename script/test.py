import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import PILToTensor, Resize, ToTensor, transforms, Normalize
import matplotlib.pyplot as plt
image_path = f"D:\datasety\cs_agent_dataset\CSGO TRAIN YOLO V5.v5i.voc\\test\csgo1628633620560459100_png_jpg.rf.37b87c8c553cf0e47140f62cec0f35bb.jpg"

transform= transforms.Compose([PILToTensor()])
transform2= transforms.Compose([])
image = Image.open(image_path)
tensor =transform(image)
tensor = tensor.type(torch.float32)
tensor=transform2(tensor)


image_np = np.asarray(image)

# Plot the image using Matplotlib
plt.imshow(image_np)
plt.show()
# Convert tensor to numpy array
tensor_np= tensor.numpy()
tensor_2d = np.reshape(tensor_np, (640, 640, 3))
# Plot the tensor using Matplotlib
plt.imshow(tensor_2d)
plt.show()

x =1




def plot_image_with_boxes(image, boxes):
    # create a copy of the image
    image = image.copy()
    draw = ImageDraw.Draw(image)

    # draw the bounding boxes on the image
    print(len(boxes[0]['boxes']))
    x=boxes[0]['boxes']
    for i in len(range(x)):
        xmin, ymin, xmax, ymax=x[i]
        draw.rectangle((xmin, ymin, xmax, ymax), outline='red')

    # display the image
    image.show()

