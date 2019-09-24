import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


def load_img():
    img= np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,text="RANDO",org=(50,300),fontFace=font,fontScale=5,color=(255,255,255),thickness=10)
    return img


def show_pic(image):
    fig =plt.figure(figsize=(12,10))
    ax=fig.add_subplot(111)
    ax.imshow(image,cmap="gray")


image = load_img()
show_pic(image)
kernel= np.ones((5,5),dtype=np.uint8)

result=cv2.erode(image, kernel=kernel,iterations=2)
show_pic(result)

white_noise= np.random.randint(low=0, high=2, size=(600,600))
white_noise = white_noise*255
show_pic(white_noise)
noise_image = white_noise + image
show_pic(noise_image)

opening = cv2.morphologyEx(noise_image,cv2.MORPH_OPEN,kernel)
show_pic(opening)

black_noise = np.random.randint(low=0, high=2, size=(600,600))
#show_pic(black_noise)
black_noise=black_noise*-255
black_noise=black_noise + image
black_noise[black_noise==-255]=0
 show_pic(black_noise)

closing = cv2.morphologyEx(black_noise, cv2.MORPH_CLOSE, kernel, iterations=2)
show_pic(closing)
show_pic(image)
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
show_pic(gradient)


image_grad = cv2.imread(r"D:\courses\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\sudoku.jpg",0)
show_pic(image_grad)
sobel_x=cv2.Sobel(image_grad, cv2.CV_64F , 1, 0, ksize=5)
show_pic(sobel_x)
sobel_y=cv2.Sobel(image_grad, cv2.CV_64F , 0, 1, ksize=5)
show_pic(sobel_y)
laplacian = cv2.Laplacian(image_grad, cv2.CV_64F)
show_pic(laplacian)
blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y, beta=0.5, gamma=0)
show_pic(blended)
ret,th1= cv2.threshold(image_grad,100,255,cv2.THRESH_BINARY)
show_pic(th1)

kernel = np.ones((4,4),np.uint8)
gradient1= cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
show_pic(gradient1)
