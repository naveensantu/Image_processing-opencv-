import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def load_img():
    img = cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\bricks.jpg").astype(np.float32)/255
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

load_img()

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)

i = load_img()
display_img(i)
gamma=np.power(i,1)
display_img(gamma)

j= load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(j, text="text checking", org=(10,600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)
kernel = np.ones(shape=(5,5),dtype=np.float32)/25
kernel
dst= cv2.filter2D(j, -1, kernel=kernel)
display_img(dst)

j= load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(j, text="text checking", org=(10,600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)

display_img(cv2.GaussianBlur(j,(15,15),10))
j= load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(j, text="text checking", org=(10,600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)

display_img(cv2.medianBlur(j, ksize=5, dst=None))

q = cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\sammy.jpg").astype(np.float32)/255
q = cv2.cvtColor(q,cv2.COLOR_BGR2RGB)
display_img(q)
noise_q = cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\sammy_noise.jpg").astype(np.float32)/255
noise_q = cv2.cvtColor(noise_q,cv2.COLOR_BGR2RGB)
display_img(noise_q)
median = cv2.medianBlur(noise_q, 5)
display_img(median)
j= load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(j, text="text checking", org=(10,600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)

display_img(cv2.bilateralFilter(j, 9, 75, 75))
