import cv2


import cv2
 
import os 


from os import listdir
from os.path import isfile, join
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


for f in names_of_big_sizes:
    fname = 'cs_agent\\images\\train\\'+f# or depending on situation: f.rstrip('\n')
    # or, if you get rid of os.chdir(path) above,
    # fname = os.path.join(path, f.rstrip())
    if os.path.isfile(fname): # this makes the code more robust
        os.remove(fname)
onlyfiles = [f for f in listdir('cs_agent\\images\\train') if isfile(join('cs_agent\\images\\train', f))]
print('size',len(onlyfiles))





import csv
with open('cs_agent\\images\\test_labels.csv', 'rb') as inp, open('cs_agent\\images\\test_labels.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[0] != onlyfiles:
            writer.writerow(row)