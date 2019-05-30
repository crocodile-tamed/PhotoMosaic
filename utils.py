# import math
import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# *********************************
# * funstions irrelative to steps *
# *********************************


def imshow(img, title='', isend=True):
    screen_res = 1280, 720
    scale_width = screen_res[0]/img.shape[1]
    scale_height = screen_res[1]/img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, window_width, window_height)
    cv2.imshow(title, img)
    if isend:
        k = cv2.waitKey(0)
        if k:
            cv2.destroyAllWindows()

def imwrite(res, outname):
    cv2.imwrite("./Result/"+outname, res, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
