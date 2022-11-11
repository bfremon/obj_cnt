#!/usr/bin/env python3

import cv2 as cv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from Log import *

set_log_lvl(True)
set_dbg_lvl(True)

img_path = os.path.join(os.getcwd(), 'img')


def usage():
    sys.stdout.write('%s input_dir output_dir\n' % (sys.argv[0]))

    
def find_img_contours(input_path,
                      out_prefix,
                      output_path = None):
    ''' 
    Find contours for image pointed by input_path
    if out_prefix is None, output is written to current dir
    out_prefix is the non optional output prefix string
    '''
    bs = os.path.basename(input_path)
    fname = bs.split('.')[:-1]
    ext = bs.split('.')[-1]
    dbg(fname, ext)
    if out_prefix is None:
        raise SyntaxError('out_prefix needed')
    filename = '%s%s.%s' % (fname, out_prefix, ext)
    if output_path is None:
        out_path = os.path.join(os.getcwd(), filename)
    else:
        out_path = os.path.join(output_path, filename)

    im = cv.imread(input_path)
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #cv.imwrite(out_prefix + 'grey.png', im_gray)
    blur_dim = 21
    im_blur = cv.GaussianBlur(im_gray, (blur_dim, blur_dim), 0)
    #cv.imwrite(out_prefix + 'blur.png', im_blur)
    ret, thres_bin = cv.threshold(im_blur, 120, 255, cv.THRESH_BINARY)
    #cv.imwrite(out_prefix + 'thres.png', thres_bin)
    contours, hierarchy = cv.findContours(thres_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    im_contour = cv.drawContours(im, contours, -1, (0, 255, 0), 3)
    cv.imwrite(out_path, im_contour)
    log('%s processed as %s' % (input_path, output_path))

    
if len(sys.argv) != 3:
    usage()
    sys.exit(-1)
else:
    input_path = os.path.join(sys.argv[1])
    output_path = os.path.join(sys.argv[1])
    for f in os.listdir(input_path):
        bs = os.path.basename(f)
        ext = bs.split('.')[-1]
        if ext == 'jpg':
            in_path = os.path.join(input_path, f)
            find_img_contours(in_path, '-contour')
