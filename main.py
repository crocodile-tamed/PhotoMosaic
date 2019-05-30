import math
import heapq
from time import time
import cv2
import numpy as np
from utils import *
import photoblocks as pb


DIR = "./RawImage/"
OUTNAME = 'result_to_step2_zz.jpg'
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
    print(b_rows, b_cols)
    bg_img = cv2.resize(bg_img, (B_SIZE*b_cols, B_SIZE*b_rows)) # round to B_SIZE*n

    blocks = [] #blocks[bx in row ][by in col]

    for _bx in range(b_rows):
        for _by in range(b_cols):
            new_block = pb.Block(_bx, _by, \
                bg_img[B_SIZE*_bx:B_SIZE*(_bx+1), B_SIZE*_by:B_SIZE*(_by+1), :])
            blocks.append(new_block)
    #blocks[bx in row ][by in col] = blocks[bx*b_cols + by]

    element_photos = []
    for _p in range(1, PHOTO_MAX):
        path = dir_ + str(_p) + ".jpg"
        img = cv2.imread(path)
        if img is not None:
            element_photos.append(pb.CmpPhoto(img, _p))
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

if __name__ == '__main__':
    T = time()
    BG_IMG = cv2.imread(DIR + 'BG.jpg')
    imshow(BG_IMG, 'original background', 0)
    RES = main(BG_IMG, DIR)
    print('time:', time()-T)
    imshow(RES, 'result', 1)
    # imwrite(RES, OUTNAME)
