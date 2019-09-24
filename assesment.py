import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
# %%
def display_img(img,cmap=None):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
# **TASK: Open and display the giaraffes.jpg image that is located in the DATA folder.**
giraffie = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\giraffes.jpg")
show_giraffie = cv2.cvtColor(giraffie, cv2.COLOR_BGR2RGB)
display_img(show_giraffie)

# **TASK:Apply a binary threshold onto the image.**

BW_giraffie = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\giraffes.jpg",0)
display_img(BW_giraffie, cmap="gray")
ret,th1=cv2.threshold(BW_giraffie, 125, 255, cv2.THRESH_BINARY)
display_img(th1,cmap="gray")


# **TASK: Open the giaraffes.jpg file from the DATA folder and convert its colorspace to  HSV and display the image.**

hsv_giraffie = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\giraffes.jpg")
show_hsv=cv2.cvtColor(hsv_giraffie,cv2.COLOR_BGR2HSV)
display_img(show_hsv)


# **TASK: Create a low pass filter with a 4 by 4 Kernel filled with values of 1/10 (0.01) and then use 2-D Convolution to blur the giraffer image (displayed in normal RGB)**

kernel=np.ones([4,4], dtype=np.float32)/10
blur= cv2.filter2D(show_giraffie, -1, kernel)
display_img(blur)

# **TASK: Create a Horizontal Sobel Filter (sobelx from our lecture) with a kernel size of 5 to the grayscale version of the giaraffes image and then display the resulting gradient filtered version of the image.**
BW_giraffie = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\giraffes.jpg",0)
display_img(BW_giraffie, cmap="gray")
x_sobel=cv2.Sobel(BW_giraffie, cv2.CV_64F, 1, 0, ksize=5)
display_img(x_sobel, cmap="gray")

# **TASK: Plot the color histograms for the RED, BLUE, and GREEN channel of the giaraffe image. Pay careful attention to the ordering of the channels.**
giraffie_for_hist= cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\giraffes.jpg")

color = ['b','g','r']
for i,col in enumerate(color):
    hist = cv2.calcHist([giraffie_for_hist], [i], None, [256], [0,256])
    plt.plot(hist,color=)

hist1 = cv2.calcHist([giraffie_for_hist], [0], None, [256], [0,256])
plt.plot(hist1,color="blue")

# %% markdown
# # Great job!
