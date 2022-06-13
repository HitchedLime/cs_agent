import cv2


import cv2
 
import os 


from os import listdir
from os.path import isfile, join
import pandas as pd 
onlyfiles = [f for f in listdir('cs_agent\\images\\train') if isfile(join('cs_agent\\images\\train', f))]

names_of_big_sizes=[]
images_diff=0
for image in  onlyfiles:
  
    img = cv2.imread('cs_agent\\images\\train\\'+image, cv2.IMREAD_UNCHANGED)
    dimensions = img.shape
    if((300, 535, 3)!=dimensions):
        images_diff+=1
        names_of_big_sizes.append(image)
    

    print('Image Dimensions :', img.shape)
print("diff images",images_diff)
print('size',len(onlyfiles))
print("big files ",names_of_big_sizes)








  
