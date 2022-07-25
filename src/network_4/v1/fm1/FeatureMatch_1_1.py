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

class FeatureMatch(nn.Module):
    def __init__(self):
        super(FeatureMatch, self).__init__()

        self.vgg16 = vgg11(pretrained=True)
        self.feature = self.vgg16.features[0:6]
        # self.feature1 = self.vgg16.features[0:6]
        # self.feature1 = self.vgg16.features[0:6]
        self.feature[0].padding = (0,0)
        self.feature[3].padding = (0,0)



    def forward(self,img,template1,template2):
        x = self.feature(img)
        kernel1 = self.feature(template1)
        kernel2 = self.feature(template2)
        out1 = self.xcorr_depthwise(x,kernel1)
        out2 = self.xcorr_depthwise(x,kernel2)
        return out1,out2

    def xcorr_depthwise(self,x,kernel):  # 对每一张图片都有单独的滤波器，所以一共要执行batchsize次滤波操作。
        # 输入通道不变，但batchsize设置为1
        out = None
        for i in range(x.size(0)):
            x_i = x[i,:].unsqueeze(0)
            kernel_i = kernel[i,:].unsqueeze(0)
            in_channel = x_i.size(1)
            out_channel = in_channel
            group = in_channel
            # x = x.view(x.size(0),batch*channel,x.size(2),x.size(3))
            kernel_i = kernel_i.view(out_channel,int(in_channel/group),kernel_i.size(2),kernel_i.size(3))
            if out is None:
                out = F.conv2d(x_i,kernel_i,groups=group)
            else:
                out = torch.cat([out,F.conv2d(x_i,kernel_i,groups=group)],0)
            # out = out.view(batch,channel,out.size(2),out.size(3))
        return out
    # def xcorr_depthwise(self,x,kernel):
    #     in_channel = x.size(1)
    #     out_channel = in_channel
    #     group = in_channel
    #     # x = x.view(x.size(0),batch*channel,x.size(2),x.size(3))
    #     kernel = kernel.view(out_channel,int(in_channel/group),kernel.size(2),kernel.size(3))
    #     out = F.conv2d(x,kernel,groups=group)
    #     # out = out.view(batch,channel,out.size(2),out.size(3))
    #     return out


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
#
# if __name__ == '__main__':
#
#     input_transform = transforms.Compose([
#         # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
#         transforms.ToPILImage(),
#         transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
#         # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
#     ])
#
#     net = FeatureMatch()
#     path = '../../../../dataset/Train/fundus_image/H0001.jpg'
#     origin = cv2.imread(path)  # 训练图像
#     template = cv2.imread('../../temolate_200_142.jpg')  # 查询图像
#
#     x = input_transform(cv2.resize(origin,dsize=(732,732))).unsqueeze(0)
#     x = x[:,:,110:622,110:622]
#     template = input_transform(cv2.resize(template,dsize=(54,54))).unsqueeze(0)
#     out,x,kernel = net(x,template)
#
#     out = out.squeeze()
#     res = add(out,115,115)
#
#     res1 = minMaxNormalization(res)
#     cv2.imwrite('./featureMate_H01_MinMaxN.png', res1)
#
#     res2 = zScore(res)
#     cv2.imwrite('./featureMate_H01_zScore.png', res2)
#
#     res_x = add(x.squeeze(),126,126)
#     res_k = add(kernel.squeeze(),12,12)
#
#     res_x_img = minMaxNormalization(res_x)
#     res_k_img = minMaxNormalization(res_k)
#
#     cv2.imwrite('./res_x_H01.png', res_x_img)
#     cv2.imwrite('./res_k_N0142.png', res_k_img)
#
#
#
#
#
#
#
#
