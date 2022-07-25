import os
import sys

import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch.nn as nn
from torchvision.models import vgg11
import torch.nn.functional as F
from torchvision import transforms

import cv2
import numpy as np
'''
【基本模型简述】将整个基本网络模型分为三个部分。
1. 第一个部分为图片处理网络，设计为多层卷积神经网络（初步定为resnet18），使用GAP将图像转为一维张量。最终将每一张图片映射为 一个 长度为 1024 的一维张量。
2. 第二个部分为文本处理网络，设计为一个简单的 全连接网络 ，最终将一条文本数据映射为一个长度为 1024 的张量。（假的图片和文本同等重要）
3. 第三个部分为一个全连接网络，输入数据为长度为 2048 的张量（将第一部分和第二部分的结果拼接，或者叠加（权重控制））， 最终将输入转为 长度为 12 的一维张量（最终预测结果）。
'''
class doubleNet_1_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(doubleNet_1_2, self).__init__()

        self.vgg16 = vgg11(pretrained=True)
        self.feature = self.vgg16.features[0:6]
        print(self.feature)

    def forward(self,img,template):
        x = self.feature(img)
        kernel = self.feature(template)
        out = self.xcorr_depthwise(x,kernel)
        return out,x,kernel

    def xcorr_depthwise(self,x,kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1,batch*channel,x.size(2),x.size(3))
        kernel = kernel.view(batch*channel,1,kernel.size(2),kernel.size(3))
        out = F.conv2d(x,kernel,groups=batch*channel)
        out = out.view(batch,channel,out.size(2),out.size(3))
        return out


def minMaxNormalization(res):
    min = res.min()
    max = res.max()
    res = (res - min) / (max - min)
    res = res * 255
    res.astype(np.int)
    return res

def zScore(matrix):
    mean = matrix.mean()
    std = matrix.std()
    matrix = (matrix-mean)/std
    matrix = matrix * 255
    matrix[matrix<0] = 0
    matrix[matrix>255] = 255
    return matrix.astype(np.int)


def add(out,H,W):
    res = torch.zeros(size=(H,W))
    for i in range(out.size(0)):
        res = res + out[i,:]
    return res.detach().numpy()

if __name__ == '__main__':

    input_transform = transforms.Compose([
        # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
    ])

    net = doubleNet_1_2(3,3)
    path = '../../../../dataset/Train/fundus_image/H0001.jpg'
    origin = cv2.imread(path)  # 训练图像
    template = cv2.imread('../../temolate_300_142.jpg')  # 查询图像

    x = input_transform(cv2.resize(origin,dsize=(512,512))).unsqueeze(0)
    template = input_transform(cv2.resize(template,dsize=(48,48))).unsqueeze(0)
    out,x,kernel = net(x,template)

    out = out.squeeze()
    res = add(out,117,117)

    res1 = minMaxNormalization(res)
    cv2.imwrite('featureMate_H01_MinMaxN.png', res1)

    res2 = zScore(res)
    cv2.imwrite('featureMate_H01_zScore.png', res2)

    res_x = add(x.squeeze(),128,128)
    res_k = add(kernel.squeeze(),12,12)

    res_x_img = minMaxNormalization(res_x)
    res_k_img = minMaxNormalization(res_k)

    cv2.imwrite('res_x_H01.png', res_x_img)
    cv2.imwrite('res_k_N0142.png', res_k_img)








