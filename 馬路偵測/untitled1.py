import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
#讀取照片
img=cv2.imread("road.jpg")
img=cv2.resize(img, (0,0),fx=1,fy=1)

#轉灰階
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#sobel
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

#合併XY
dst = cv2.sqrt(sobel_x**2+sobel_y**2)

#設定LBP偵測
radius = 1
n_points = 8 * radius
lbp=local_binary_pattern(dst,n_points, radius,method="default")
lbp_flattened = lbp.ravel()
#把值丟進histogram
n_bins = int(lbp.max() + 1)  
hist, bins = np.histogram(lbp_flattened, bins=n_bins, range=(0, n_bins), density=True)
plt.figure(figsize=(10, 5))

#search
#31~39還在測試階段(一個找查單一點、另一個利用某個點的紋理扣除另一點紋理)


#road_texture_min =31
#road_texture_max =31
road_texture_min =(lbp==30)
road_texture_max =(lbp==31)

#mask = np.logical_and(lbp >= road_texture_min, lbp <= road_texture_max)
mask = np.logical_and(road_texture_min,np.logical_not(road_texture_max))
mask_uint8 = mask.astype(np.uint8) * 255
#histogram表值設定
plt.bar(bins[:-1], hist, width=0.8, color='gray')
plt.title("LBP Histogram")
plt.xlabel("LBP value")
plt.ylabel("Frequency")
plt.show()
cv2.imshow('img',mask_uint8 )

#(測試用)有無抓到LBP數值及範圍
print("LBP 最小值:", np.min(lbp))
print("LBP 最大值:", np.max(lbp))
cv2.waitKey(0)

#目前search找不到適合的點，後續考慮用高斯模糊後再用sobel去找尋
#找尋到後將檢測到的點上色並進行與圖片合併
