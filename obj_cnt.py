#!/usr/bin/env python3

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img_path = os.path.join(os.getcwd(), 'img')

im = cv2.imread(os.path.join(img_path, 'test.png'))
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite('test-grey.png', im_gray)
blur_dim = 21
im_blur = cv2.GaussianBlur(im_gray, (blur_dim, blur_dim), 0)
cv2.imwrite('test-blur.png', im_blur)
canny_hyst_lsl = 1
canny_hyst_usl = 10
im_cont = cv2.Canny(im_blur, canny_hyst_lsl, canny_hyst_usl, 3)
cv2.imwrite('test-cont.png', im_cont)
im_dilated = cv2.dilate(im_cont, (25, 25), iterations = 1)
cv2.imwrite('test-dil.png', im_dilated)

(cnt, hierarchy) = cv2.findContours(
    im_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
cv2.drawContours(im_rgb, cnt, -1, (0, 255, 0), 2)
cv2.imwrite('test-cnt.png', im_rgb)
print(len(cnt))
