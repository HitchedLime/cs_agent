from PIL import  Image

from transforms.transforms_custom import plot_image_with_boxes

img = Image.open(f"D:\datasety\cs_agent_dataset\\train\images\csgo1628633666963932100_png_jpg.rf.8f944584e7258145ca8b9228b87e45d8.jpg")
target ={"bbox":[[1, 0.49609375, 0.52578125, 0.0359375, 0.0828125]]}
#xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]


plot_image_with_boxes(img, target["bbox"])


target["bbox"]=1
print(target)

