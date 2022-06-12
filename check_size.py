import cv2


import cv2
 



from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("images\\train") if isfile(join("images\\train", f))]
# read image

images_diff=0
for image in  onlyfiles:
  
    img = cv2.imread('C:\\Users\\david\\Desktop\\cs_agent\\images\\train\\'+image, cv2.IMREAD_UNCHANGED)
    dimensions = img.shape
    if((300, 535, 3)!=dimensions):
        images_diff+=1
    

    print('Image Dimensions :', img.shape)
print("diff images",images_diff)
print('size',len(onlyfiles))