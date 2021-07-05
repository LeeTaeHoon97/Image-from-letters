import cv2,os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image


"""
   #Inspired by https://www.kaggle.com/anokas/kuzushiji-mnist-cat
   
    #image resize

import imutils
path_to_files = "letter/"
result_path="resized_letter/"
for _, file in enumerate(os.listdir(path_to_files)):
    img=cv2.imread(path_to_files+file)
    img = imutils.resize(img,25,25)
    cv2.imwrite(result_path+file, img)
"""
"""
#convert letter images  to npz file




from PIL import Image
path_to_files = "resized_letter/"
array_of_images = []

for _, file in enumerate(os.listdir(path_to_files)):
    single_im = Image.open(path_to_files+file)
    single_array = np.array(single_im)
    array_of_images.append(single_array) 
    
np.savez("letter_data.npz",array_of_images) # save all in one file

"""
#Load and resize image

img_path="image/test.jpg"

img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

img=cv2.resize(img,dsize=None,fx=0.2,fy=0.2)


plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(img,cmap='gray')

"""
if error "ValueError: Object arrays cannot be loaded when allow_pickle=False" occured,
type this code.

np.load.__defaults__=(None, True, True, 'ASCII')
"""

sample_imgs=np.load('letter_data.npz')['arr_0']


#show letters
plt.figure(figsize=(20,10))
for i in range(80):
    img_patch=sample_imgs[i]

    plt.subplot(5,16,i+1)
    plt.title(int(np.mean(img_patch)))
    plt.axis('off')
    plt.imshow(img_patch,cmap='gray')



#Summary of letter images

means= np.mean(sample_imgs,axis=(1,2,3))

plt.figure(figsize=(12,6))
plt.hist(means,bins=50,log=True)
plt.show()

img=cv2.normalize(img,dst=None,alpha=180,beta=230,norm_type=cv2.NORM_MINMAX)
plt.figure(figsize=(12,6))
plt.hist(img.flatten(),bins=50,log=True)
plt.show()

#organize images
bins=defaultdict(list)

for img_patch,mean in zip(sample_imgs,means):
    bins[int(mean)].append(img_patch)

print(len(bins))

#fill image

h,w = img.shape
img_out = np.zeros((h*25,w*25,),dtype=np.uint8)

for y in range(h):
    for x in range(w):
        
        pixel = img[y,x]

        b=bins[pixel]
        while len(b)==0:
            pixel+=1
            b=bins[pixel]

        img_patch=b[np.random.randint(len(b))][:, :, 0]

        img_out[y*25:(y+1)*25,x*25:(x+1)*25]=img_patch


plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(img_out,cmap='gray')


cv2.imwrite("result/"+'result.jpeg',img_out)