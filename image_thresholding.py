import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


image=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\rainbow.jpg",0)
plt.imshow(image,cmap="gray")
ret,thresh1=cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
ret
plt.imshow(thresh1,cmap="gray")
new_image=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\crossword.jpg",0)
plt.imshow(new_image,cmap="gray")

def show_pic(image):
    fig =plt.figure(figsize=(25,25))
    ax=fig.add_subplot(111)
    ax.imshow(image,cmap="gray")

#show_pic(new_image)

ret,th1=cv2.threshold(new_image, 180, 255,cv2.THRESH_BINARY)
show_pic(th1)
th2 = cv2.adaptiveThreshold(new_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(th2)
blended = cv2.addWeighted(src1=th1, alpha=0.6, src2=th2, beta=0.4, gamma=0)
show_pic(blended)
print("making changes")
