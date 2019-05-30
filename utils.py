# import math
import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# *********************************
# * funstions irrelative to steps *
# *********************************

class ArgError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo
    def __str__(self):
        return self.errorinfo

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
    if '.jpg' in outname or '.jpg' in outname:
        cv2.imwrite("./Result/"+outname, res, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    else:
        cv2.imwrite("./Result/"+outname, res, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

def check_arg(opt, arg):
    if opt == 'd':
        if arg[-1] != '/':
            arg += '/'
        return arg
    elif opt == 'o':
        if '.jpg' not in arg and '.jpeg' not in arg and '.png' not in arg:
            print('-o: will output .jpg file as default, please specify file format')
        return arg+'.jpg'
    elif opt == 'v':
        if opt in "NonoFalsefalse0":
            return False
        else:
            return True
    elif opt == 'b':
        try:
            arg = int(arg)
        except ValueError:
            raise ArgError('-b: please input block size in integer')
        return arg
    elif opt == 'm':
        try:
            arg = int(arg)
        except ValueError:
            raise ArgError('-m: please input max photo index in integer')
        return arg
    elif opt == 'c':
        try:
            arg = int(arg)
        except ValueError:
            raise ArgError('-c: please input check depth in integer')
        return arg
    elif opt == 'i':
        if '.jpg' not in arg and '.jpeg' not in arg and '.png' not in arg:
            raise ArgError('-i: please input a proper target image name')
        else:
            return arg
HELP_INFO = """
help informaion:
-d --dir            The input directory for both souurces and target image. Default: ./RawImage
-i --input          The target file. File should be in the dir.             Default: BG(0).jpg
-o --output         The output file name, better end in .jpg/.jpeg/.png.    Default: by default will not save.
-v --visual         Whether to show the images.                             Default: True
-b --blocksize      Specify the block size (of one-side).                   Default: 64
-m --photomax       Specify the max index of material files.                Default: 300
-c --checkdepth     Specify the check depth for the redundancy removal.     Default: 3
"""
