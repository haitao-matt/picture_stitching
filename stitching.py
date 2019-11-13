# -*- coding:utf-8 -*-
'''
date: 2019-11-13
auth: matt
function: stitch picture
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt


def switch_channel(img):
    R, G, B = cv2.split(img)
    img = cv2.merge((B, G, R))
    return img

def show_original(img1, img2):
    plt.ion()
    img1 = switch_channel(img1)
    img2 = switch_channel(img2)

    plt.subplot(121), plt.imshow(img2),plt.title('Original pitcture img2')
    plt.axis('off')
    plt.subplot(122), plt.imshow(img1),plt.title('Original pitctures img1')
    plt.axis('off')
    plt.pause(5)
    plt.close()


def get_kps_des(img1, img2):
    '''
    @ get picture key point & descriptor
    :param img1:
    :param img2:
    :return: kp1, des1, kp2, des2, good
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    '''
    Keypoints between two images are matched by identifying their nearest neighbours. But in some cases, 
    the second closest-match may be very near to the first. It may happen due to noise or some other reasons. 
    In that case, ratio of closest-distance to second-closest distance is taken. If it is greater than 0.8, they are rejected. 
    It eliminaters around 90% of false matches while discards only 5% correct matches, as per the paper.
    '''
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: # we use 0.75
            good.append([m])
    return kp1, des1, kp2, des2, good

def draw_match(match_img):
    match_img = switch_channel(match_img)
    plt.ion()
    plt.imshow(match_img), plt.title('match_img')
    plt.axis('off')
    plt.pause(5)
    plt.ioff()
    plt.close()


def stitch_picture(img1, img2, kp1, kp2, good):
    # img1 descriptor indexes
    src_pts = np.array([kp1[m[0].queryIdx].pt for m in good])
    # img2 descriptor indexes
    dst_pts = np.array([kp2[m[0].trainIdx].pt for m in good])

    # get homegraphy matrix utilizing RANSAN method
    H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # perspective transformation
    dst_corners = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
    # merge the second picture on the right side
    dst_corners[0:h2, 0:w2] = img2

    # draw picture
    R, G, B = cv2.split(dst_corners)
    img_match = cv2.merge((B, G, R))
    plt.imshow(img_match)
    plt.axis('off')
    plt.title('Stitching picture')
    plt.show()

if __name__ == "__main__":
    # img1 right picture
    img1 = cv2.imread(r'./image/picture01.jpg')
    # img2 left picture
    img2 = cv2.imread(r'./image/picture02.jpg')
    show_original(img1, img2)
    # plt.imshow(img1),plt.show()
    kp1, des1, kp2, des2, good = get_kps_des(img1, img2)
    match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    draw_match(match_img)
    stitch_picture(img1, img2, kp1, kp2, good)
