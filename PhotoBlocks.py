import math
import numpy as np
import cv2

class Block:
    '''class for blocks in image(x,y)'''

    def __init__(self, pos_x, pos_y, img):
        self.pos = [pos_x, pos_y]
        self.img = img #64*64*3
        self.partition() # get areas and feature values
        self.photo_match_id = None # idx in photo_ids
        self.photo_ids = None # store 20 smallest photo's idx in element string
        self.photo_dis = None # store the corresponding distances to photo_ids

    def partition(self): #BGR 4*4 * 3 each 16*16
        self.features = np.array([[[0 for _ in range(3)] for _ in range(4)] for _ in range(4)])
        for _i in range(4):
            for _j in range(4):
                # self.cal_feature_value(_i, _j)
                segment = self.img[_i*16:(_i+1)*16, _j*16:(_j+1)*16, :]
                self.features[_i, _j, 0] = np.mean(segment[..., 0])
                self.features[_i, _j, 1] = np.mean(segment[..., 1])
                self.features[_i, _j, 2] = np.mean(segment[..., 2])

    # def cal_feature_value(self, _i, _j):
    #     segment = self.img[_i*16:(_i+1)*16, _j*16:(_j+1)*16, :]
    #     self.features[_i, _j, 0] = np.mean(segment[..., 0])
    #     self.features[_i, _j, 1] = np.mean(segment[..., 1])
    #     self.features[_i, _j, 2] = np.mean(segment[..., 2])

    def update_photo(self, photoid, ids, dis):
        self.photo_match_id = photoid
        self.photo_ids = ids
        self.photo_dis = dis

    def matchid(self):
        return self.photo_ids[self.photo_match_id]

    def matchdis(self):
        return self.photo_dis[self.photo_match_id]

    def imshow(self):
        title = 'block(' + str(self.pos[0]) + ', ' + str(self.pos[1]) + ')'
        cv2.imshow(title, self.img)

# distance between original block & compared block
def difference(ori_b, cmp_b):
    return np.sum((ori_b-cmp_b)**2)

class CmpPhoto:
    def __init__(self, img, id_):
        self.img = cv2.resize(img, (64, 64))
        self.id_ = id_
        self.partition()

    def partition(self):
        self.features = np.array([[[0 for _ in range(3)] for _ in range(4)] for _ in range(4)])
        [rows, cols, _] = self.img.shape
        blocks_r = [0, math.ceil(0.25*rows), math.ceil(0.5*rows), math.ceil(0.75*rows), rows]
        blocks_c = [0, math.ceil(0.25*cols), math.ceil(0.5*cols), math.ceil(0.75*cols), cols]
        for _i in range(4):
            for _j in range(4):
                segment = self.img[blocks_r[_i]:blocks_r[_i+1], blocks_c[_j]:blocks_c[_j+1], :]
                self.features[_i, _j, 0] = np.mean(segment[..., 0])
                self.features[_i, _j, 1] = np.mean(segment[..., 1])
                self.features[_i, _j, 2] = np.mean(segment[..., 2])

    def imshow(self, title=''):
        title = 'element' + str(title)
        cv2.imshow(title, self.img)


def check_redundancy(blocks, pos, depth, shape):
    def get_id(pos):
        return shape[1]*pos[0] + pos[1]

    block = blocks[get_id(pos)]
    #the check area
    for _r in range(pos[0]-depth, pos[0]+1):
        for _c in range(pos[1]-depth, pos[1]+1+pos[0]-_r):
            if _r < 0 or _c < 0 or _r > shape[0] or _c > shape[1] or [_r, _c] == pos:
                continue
            b_id = get_id([_r, _c])
            # print(_r, _c, b_id, pos)
            if blocks[b_id].matchid() == block.matchid():
                block.photo_match_id = (block.photo_match_id + 1) % len(block.photo_ids)
                # if blocks[b_id].matchdis() < block.matchdis():
                #     blocks[b_id].photo_match_id = \
                #         (blocks[b_id].photo_match_id + 1) % len(blocks[b_id].photo_ids)
                # else:
                #     block.photo_match_id = (block.photo_match_id + 1) % len(block.photo_ids)
