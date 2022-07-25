# 数据增强
# 1.生成 0-1 之间的随机数，当随机数大于0.6的时候进行数据增强（加入高斯噪声）
# 2.高斯噪声的均值为0，方差为 0.01 到 0.05 之间的随机数


import cv2
import torch

img = cv2.imread('20051020_57844_0100_PP.tif')


import random
#
# random.gauss()

import numpy as np

for i in range(10000):
    p = np.random.rand(1)
    print('p={}'.format(p))
    if p >=0.6:
        q = np.random.uniform(0.01,0.05,1)
        print('q = {}'.format(q))
        noise = np.random.normal(0,q,size=(5,5)).astype(np.float32)
        noise = torch.from_numpy(noise)
        print(noise)

