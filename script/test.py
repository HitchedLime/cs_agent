from PIL import  Image

from transforms.transforms_custom import plot_image_with_boxes

img = Image.open(f"D:\datasety\cs_agent_dataset\\train\csgo1628633666963932100_png_jpg.rf.8f944584e7258145ca8b9228b87e45d8.jpg")
target ={"bbox":[[307, 330, 311, 364]]}
#


plot_image_with_boxes(img, target["bbox"])


target["bbox"]=1
print(target)

