import math
from time import time
import cv2
import numpy as np
from utils import *
import PhotoBlocks as pb

DIR = "./RawImage/"
OUTNAME = 'result_to_step2.jpg'
BLOCK_SIZE = 64
# COLOR_DIFF_THRE = 1
PHOTO_MAX = 300

def main(bg_img, dir_):

    # ****************************************
    # * STEP1 Partition + Feature Extraction *
    # ****************************************

    ### round to 64*n
    [bg_rows, bg_cols, _] = bg_img.shape #type = np.ndarray(rols,cols,3)
    b_rows = math.ceil(bg_rows / 64)
    b_cols = math.ceil(bg_cols / 64)
    bg_img = cv2.resize(bg_img, (64*b_cols, 64*b_rows)) # round to 64*n

    blocks = [] #blocks[bx in row ][by in col]

    for _bx in range(b_rows):
        for _by in range(b_cols):
            new_block = pb.Block(_bx, _by, bg_img[64*_bx:64*(_bx+1), 64*_by:64*(_by+1), :])
            blocks.append(new_block)
    del _bx
    del _by
    #blocks[bx in row ][by in col] = blocks[bx*b_rows + by]

    element_photos = []
    for _p in range(1, PHOTO_MAX):
        path = dir_ + str(_p) + ".jpg"
        img = cv2.imread(path)
        if img is not None:
            element_photos.append(pb.CmpPhoto(img))
        # if (_p % 100) == 1:
        #     print(element_photos[-1].features)

    # ************************
    # * STEP2 Block Matching *
    # ************************

    for block in blocks:
        min_diff = float('inf')
        photo_idx = 0
        for _i, ele_photo in enumerate(element_photos):
            diff = pb.difference(block.features, ele_photo.features)
            if diff < min_diff:
                min_diff = diff
                photo_idx = _i
        block.update_photo(element_photos[photo_idx])
    # *************************************
    # * STEP3 Redundancy Removal of Photo *
    # *************************************

    # **************************
    # * STEP4 Color Adjustment *
    # **************************

    # ************************
    # * STEP5 Reconstruction *
    # ************************
    output = np.array([[[0 for _ in range(3)] for _ in range(b_cols*64)] for _ in range(b_rows*64)])
    for block in blocks:
        [_row, _col] = block.pos
        # print(output[64*_row:64*(_row+1), 64*_col:64*(_col+1), :].shape)
        output[64*_row:64*(_row+1), 64*_col:64*(_col+1), :] = block.photo_match.img
    return output.astype(np.uint8)

if __name__ == '__main__':
    T = time()
    BG_IMG = cv2.imread(DIR + 'BG.jpg')
    # bg_img = cv2.cvtColor(bg_img,cv2.COLOR_BGR2RGB) # uncomment this if use PLT to show
    imshow(BG_IMG, 'original background', 0)

    RES = main(BG_IMG, DIR)
    print('time:', time()-T)
    # imshow(RES, 'result', 1)
    imwrite(RES, OUTNAME)
