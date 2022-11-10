#!/usr/bin/env python3

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img_path = os.path.join(os.getcwd(), 'img')

out_prefix = 'test-find_contour'

im = cv2.imread(os.path.join(img_path, 'test.png'))
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite(out_prefix + 'grey.png', im_gray)
blur_dim = 21
im_blur = cv2.GaussianBlur(im_gray, (blur_dim, blur_dim), 0)
cv2.imwrite(out_prefix + 'blur.png', im_blur)
ret, thres_bin = cv2.threshold(im_blur, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite(out_prefix + 'thres.png', thres_bin)
contours, hierarchy = cv2.findContours(thres_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
im_contour = cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
cv2.imwrite(out_prefix + 'contour.png', im_contour)
