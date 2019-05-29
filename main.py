import math
from time import time
# import cv2
from utils import *
from PartitionNFeatureExtraction import Block

BLOCK_SIZE = 64

def main(bg_img):

    # ****************************************
    # * STEP1 Partition + Feature Extraction *
    # ****************************************

    ### round to 64*n
    [bg_rows, bg_cols, _] = bg_img.shape #type = np.ndarray(rols,cols,3)
    b_rows = math.ceil(bg_rows / 64)
    b_cols = math.ceil(bg_cols / 64)
    bg_img = cv2.resize(bg_img, (64*b_cols, 64*b_rows)) # round to 64*n

    blocks = [[] for _ in range(b_rows)] #blocks[bx in row ][by in col]

    for _bx in range(b_rows):
        for _by in range(b_cols):
            new_block = Block(_bx, _by, bg_img[64*_bx:64*(_bx+1), 64*_by:64*(_by+1), :])
            blocks[_bx].append(new_block)

    # ************************
    # * STEP2 Block Matching *
    # ************************

    # *************************************
    # * STEP3 Redundancy Removal of Photo *
    # *************************************

    # **************************
    # * STEP4 Color Adjustment *
    # **************************

    return bg_img

if __name__ == '__main__':
    T = time()
    BG_IMG = cv2.imread('./RawImage/0.jpg')
    # bg_img = cv2.cvtColor(bg_img,cv2.COLOR_BGR2RGB) # uncomment this if use PLT to show
    imshow(BG_IMG, 'original background', 0)
    RES = main(BG_IMG)
    print('time:',time()-T)
    imshow(RES, 'res', 1)
    
