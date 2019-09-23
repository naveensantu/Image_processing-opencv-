import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline


image=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\00-puppy.jpg")
plt.imshow(image)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image)
img_hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
plt.imshow(img_hsv)
