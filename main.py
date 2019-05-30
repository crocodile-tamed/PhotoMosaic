import math
import heapq
import sys
import getopt
from time import time
import cv2
import numpy as np
from utils import *
import photoblocks as pb


DIR = "./RawImage/"
BG_NAME = "0.jpg"
OUTNAME = 'test'
IM_SHOW = True
IM_WRITE = False
B_SIZE = 64 #block size
# COLOR_DIFF_THRE = 1
PHOTO_MAX = 300
CHECK_DEPTH = 3

def main(bg_img, dir_):

    # ****************************************
    # * STEP1 Partition + Feature Extraction *
    # ****************************************

    ### round to B_SIZE*n
    [bg_rows, bg_cols, _] = bg_img.shape #type = np.ndarray(rols,cols,3)
    b_rows = math.ceil(bg_rows / B_SIZE)
    b_cols = math.ceil(bg_cols / B_SIZE)
    # print(b_rows, b_cols)
    bg_img = cv2.resize(bg_img, (B_SIZE*b_cols, B_SIZE*b_rows)) # round to B_SIZE*n

    blocks = [] #blocks[bx in row ][by in col]

    for _bx in range(b_rows):
        for _by in range(b_cols):
            new_block = pb.Block(_bx, _by, \
                bg_img[B_SIZE*_bx:B_SIZE*(_bx+1), B_SIZE*_by:B_SIZE*(_by+1), :], B_SIZE)
            blocks.append(new_block)
    #blocks[bx in row ][by in col] = blocks[bx*b_cols + by]

    element_photos = []
    for _p in range(0, PHOTO_MAX):
        path = dir_ + str(_p) + ".jpg"
        img = cv2.imread(path)
        if img is not None:
            element_photos.append(pb.CmpPhoto(img, _p, B_SIZE))
        # if (_p % 100) == 1:
        #     print(element_photos[-1].features)

    # ************************ & *************************************
    # * STEP2 Block Matching * & * STEP3   of Photo *
    # ************************ & *************************************

    for block in blocks:
        diffs = []
        for _i, ele_photo in enumerate(element_photos):
            diff = pb.difference(block.features, ele_photo.features)
            diffs.append(diff)
        photo_idx = list(map(diffs.index, heapq.nsmallest(20, diffs)))
        photo_dis = [diffs[i] for i in photo_idx]
        block.update_photo(0, photo_idx, photo_dis)
        # redundancy removal
        pb.check_redundancy(blocks, block.pos, CHECK_DEPTH, [b_rows, b_cols])

    # ************************** & ************************
    # * STEP4 Color Adjustment * & * STEP5 Reconstruction *
    # ************************** & ************************

    output = np.array([[[0, 0, 0] for _ in range(b_cols*B_SIZE)] for _ in range(b_rows*B_SIZE)])
    for block in blocks:
        [_row, _col] = block.pos
        blk_img = element_photos[block.matchid()].img
        # print(block.pos, element_photos[block.matchid()].id_)
        with np.errstate(divide='ignore', invalid='ignore'):
            # HSI mode = HLS in opencv
            blk_img = cv2.cvtColor(blk_img, cv2.COLOR_BGR2HLS)
            blk_ori = block.img
            blk_ori = cv2.cvtColor(blk_ori, cv2.COLOR_BGR2HLS)
            blk_img[:, :, 2] = blk_ori[:, :, 2]
            blk_img = cv2.cvtColor(blk_img, cv2.COLOR_HLS2BGR)
            output[B_SIZE*_row:B_SIZE*(_row+1), B_SIZE*_col:B_SIZE*(_col+1), :] \
                = blk_img

    return output.astype(np.uint8)

def deal_args():
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv, "hd:i:o:v:b:m:c:", \
            ["input=", "dir=", "output=", "visual=", "blocksize=", "photomax=", "checkdepth="])
    except getopt.GetoptError:
        global HELP_INFO
        raise ArgError(HELP_INFO)
    if opts == []:
        print("""
******************************************
* arguments available! more info with -h *
******************************************
""")
    for opt, arg in opts:
        # print(opt, arg)
        if opt in ('-d', '--dir'):
            global DIR
            DIR = check_arg('d', arg)
        elif opt in ('-i', '--input'):
            global BG_NAME
            BG_NAME = check_arg('i', arg)
        elif opt in ('-o', '--output'):
            global OUTNAME
            global IM_WRITE
            IM_WRITE = True
            OUTNAME = check_arg('o', arg)
        elif opt in ('-v', '--visual'):
            global IM_SHOW
            IM_SHOW = check_arg('v', arg)
        elif opt in ('-b', '--blocksize'):
            global B_SIZE
            B_SIZE = check_arg('b', arg)
        elif opt in ('-m', '--photomax'):
            global PHOTO_MAX
            PHOTO_MAX = check_arg('m', arg)
        elif opt in ('-c', '--checkdepth'):
            global CHECK_DEPTH
            CHECK_DEPTH = check_arg('c', arg)
        else:
            print(HELP_INFO)
            sys.exit(1)

if __name__ == '__main__':
    deal_args()
    T = time()
    BG_IMG = cv2.imread(DIR + BG_NAME)
    if BG_IMG is None:
        raise ArgError('-i: please input a proper target image name in the directory')
    if IM_SHOW:
        imshow(BG_IMG, 'original background', 0)
    RES = main(BG_IMG, DIR)
    print('time:', time()-T)
    if IM_SHOW:
        imshow(RES, 'result', 1)
    if IM_WRITE:
        imwrite(RES, OUTNAME)
