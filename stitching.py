import numpy as np
import cv2
from matplotlib import pyplot as plt
# img1 right picture
img1 = cv2.imread(r'E:\AI_Project\week2\homework\match\right.jpg')
# img2 left picture
img2 = cv2.imread(r'E:\AI_Project\week2\homework\match\left.jpg')

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)



# img1 descriptor indexes
src_pts = np.array([ kp1[m.queryIdx].pt for m in good])
# img2 descriptor indexes
dst_pts = np.array([ kp2[m.trainIdx].pt for m in good])

# get homegrapht matrix utilizing RANSAN method
H,_ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)

h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]

# perspective transformation
dst_corners=cv2.warpPerspective(img1, H, (w1 + w2,max(h1, h2)))
# merge the second picture on the right side
dst_corners[0:h2,0:w2]=img2

# draw picture
R, G, B= cv2.split(dst_corners)
img_match = cv2.merge((B, G, R))
plt.imshow(img_match),plt.show()
