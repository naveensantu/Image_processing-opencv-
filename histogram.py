import cv2
import matplotlib.pyplot as plt

import numpy as np
%matplotlib inline

dark_horse = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\horse.jpg")
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\rainbow.jpg")
show_rainbow=cv2.cvtColor(rainbow,cv2.COLOR_BGR2RGB)

blue_bricks=cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\bricks.jpg")
show_bricks=cv2.cvtColor(blue_bricks,cv2.COLOR_BGR2RGB)
plt.imshow(show_bricks)



histvalues=cv2.calcHist([blue_bricks], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(histvalues)


histvalues_horse=cv2.calcHist([dark_horse], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(histvalues_horse)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([blue_bricks], [i], None, [256], [0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])
for i,col in enumerate(color):
    histr = cv2.calcHist([dark_horse], [i], None, [256], [0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])


rainbow1 = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\rainbow.jpg")
show_rainbow=cv2.cvtColor(rainbow1,cv2.COLOR_BGR2RGB)

rainbow1.shape
mask = np.zeros(rainbow1.shape[:2],np.uint8)
plt.imshow(mask,cmap="gray")
mask[300:400,100:400]=255
plt.imshow(mask,cmap="gray")


masked_image=cv2.bitwise_and(rainbow1, rainbow1, mask=mask)
show_masked_img =cv2.bitwise_and(show_rainbow,show_rainbow,mask=mask)
plt.imshow(show_masked_img)
hist_mask_values_red= cv2.calcHist([rainbow1], channels=[2], mask=mask, histSize=[256], ranges=[0,256])
hist_mask_red= cv2.calcHist([rainbow1], channels=[2], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_mask_values_red)
plt.title("red histogram for masked rainbow")
 plt.plot(hist_mask_red)

gorilla = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\gorilla.jpg",0)
def show_pic(image):
    fig =plt.figure(figsize=(25,25))
    ax=fig.add_subplot(111)
    ax.imshow(image,cmap="gray")

show_pic(gorilla)

gorilla.shape


hist_values = cv2.calcHist([gorilla], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_values)
eq_gorilla = cv2.equalizeHist(gorilla)
show_pic(eq_gorilla)
hist_values = cv2.calcHist([eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_values)
color_gorilla = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\gorilla.jpg")
show_gorilla= cv2.cvtColor(color_gorilla,cv2.COLOR_BGR2RGB)
show_pic(color_gorilla)


hsv = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)
hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
eq_color_gorilla = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
show_pic(eq_color_gorilla)
