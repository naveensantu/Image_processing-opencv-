import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np


img1=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\dog_backpack.png")
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
plt.imshow(img1)
img2=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\watermark_no_copy.png")
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
plt.imshow(img2)
img1.shape
img2.shape

img1_resize=cv2.resize(img1,(1200,1200))
img2_resize=cv2.resize(img2,(1200,1200))


img_blended=cv2.addWeighted(src1=img1_resize, alpha=0.8,src2=img2_resize, beta=0.2, gamma=10)
plt.imshow(img_blended)

###overlay images without blending using numpy
img1_larger=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\dog_backpack.png")
img1_larger=cv2.cvtColor(img1_larger,cv2.COLOR_BGR2RGB)
plt.imshow(img1_larger)
img2_small=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\watermark_no_copy.png")
img2_small=cv2.cvtColor(img2_small,cv2.COLOR_BGR2RGB)
plt.imshow(img2_small)
img2_small=cv2.resize(img2_small,(600,600))
img2_small.shape

x_offset=100
y_offset=100

x_end= x_offset + img2_small.shape[1]
y_end=y_offset + img2_small.shape[0]

img1_larger[y_offset:y_end,x_offset:x_end]=img2_small
plt.imshow(img1_larger)

###blend together images of different sizes
img_large=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\dog_backpack.png")
img_large=cv2.cvtColor(img_large,cv2.COLOR_BGR2RGB)
img_small=cv2.imread(r"D:\Computer Vision with OpenCV and Deep Learning\Computer-Vision-with-Python\DATA\watermark_no_copy.png")
img_small=cv2.cvtColor(img_small,cv2.COLOR_BGR2RGB)
img_small=cv2.resize(img_small,(600,600))

img_large.shape


x_offset1 = 934-600
y_offset1= 1401-600

img_small.shape

rows,cols,channels = img_small.shape

roi = img_large[y_offset1:1401,x_offset1:943]
plt.imshow(roi)
img_small_gray= cv2.cvtColor(img_small,cv2.COLOR_RGB2GRAY)
plt.imshow(img_small_gray,cmap="gray")
mask_inv=cv2.bitwise_not(img_small_gray)
plt.imshow(mask_inv,cmap="gray")
mask_inv.shape
white_background = np.full(img_small.shape,255,dtype=np.uint8)

bk = cv2.bitwise_or(white_background,white_background,mask=mask_inv)
bk.shape

plt.imshow(bk)
fg=cv2.bitwise_or(img_small,img_small,mask=mask_inv)
plt.imshow(fg)

final_roi= cv2.bitwise_or(roi, fg)
plt.imshow(final_roi)
large_img = img_large
small_image=final_roi

large_img[y_offset1:y_offset1+small_image.shape[0],x_offset1:x_offset1+small_image.shape[1]]=small_image
plt.imshow(large_img)
