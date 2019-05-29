import numpy as np
import cv2

class Block:
    '''class for blocks in image(x,y)'''

    def __init__(self, pos_x, pos_y, img):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.img = img #64*64*3
        self.partition() # get areas and feature values

    def partition(self): #BGR 4*4 * 3 each 16*16
        self.features_b = np.array([[0 for _ in range(4)] for _ in range(4)])
        self.features_g = np.array([[0 for _ in range(4)] for _ in range(4)])
        self.features_r = np.array([[0 for _ in range(4)] for _ in range(4)])
        for _i in range(4):
            for _j in range(4):
                # self.cal_feature_value(_i, _j)
                segment = self.img[_i*16:(_i+1)*16, _j*16:(_j+1)*16, :]
                self.features_b[_i, _j] = np.mean(segment[..., 0])
                self.features_g[_i, _j] = np.mean(segment[..., 1])
                self.features_r[_i, _j] = np.mean(segment[..., 2])

    # def cal_feature_value(self, _i, _j):
    #     segment = self.img[_i*16:(_i+1)*16, _j*16:(_j+1)*16, :]
    #     self.features_b[_i][_j] = np.mean(segment[..., 0])
    #     self.features_g[_i][_j] = np.mean(segment[..., 1])
    #     self.features_r[_i][_j] = np.mean(segment[..., 2])

    def imshow(self):
        '''show the block'''
        title = 'block(' + str(self.pos_x) + ', ' + str(self.pos_y) + ')'
        cv2.imshow(title, self.img)
