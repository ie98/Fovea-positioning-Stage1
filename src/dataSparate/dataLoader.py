import os
import time
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import yaml
from PIL import Image
import sys
sys.path.append(os.path.abspath('../..'))
print(os.path.abspath('../..'))
yamlFilePath = './config/dataload.yaml'
with open(yamlFilePath, 'r') as f:
    cfg = yaml.safe_load(f)

root_path = cfg['root_path']

random.seed(1)

def doRandom(array):
    randomlist = random.sample(range(0, array.shape[0]), array.shape[0])
    array = array[randomlist]

    return array

def loadData(rootPath):
    train_list = np.array(pd.read_csv(os.path.join(rootPath, 'data/train_list.csv')))[:, [1, 3, 4]] # [name,x,y] = [name,col,row]
    val_list = np.array(pd.read_csv(os.path.join(rootPath, 'data/val_list.csv')))[:, [1, 3, 4]]
    test_list = np.array(pd.read_csv(os.path.join(rootPath, 'data/test_list.csv')))[:, [1, 3, 4]]

    # 打乱数据
    train_list = doRandom(train_list)
    val_list = doRandom(val_list)
    test_list = doRandom(test_list)

    return train_list,val_list,test_list


def loadImgs(img_name,label):
    downSample_W = cfg['downSample_W']
    downSample_H = cfg['downSample_H']
    new_labels = np.ndarray(shape=[1, 2])
    img = cv2.imread(os.path.join(root_path, img_name)) # 读取的是 【行，列，3】 = 【y,x,3】  注意 标注是 [x,y]

    # 计算图像下采样后，标签的位置
    rate_x = 512 / img.shape[1]
    rate_y = 512 / img.shape[0]
    new_label_x = int( label[0,0] * rate_x)
    new_label_y = int( label[0,1] * rate_y)
    new_labels[0,1] = new_label_y
    new_labels[0,0] = new_label_x

    # 图像下采样
    img.astype(np.uint8)
    img = cv2.resize(img, dsize=(downSample_H, downSample_W), interpolation=cv2.INTER_LINEAR)

    return img,new_labels.reshape(-1)

def getDataload():
    # 图片处理
    input_transform = transforms.Compose([
        # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
    ])
    train_list,val_list,test_list = loadData('./dataSparate/data')
    train_image_names = train_list[:,0]
    train_labels = train_list[:,[1,2]].astype(np.float32)
    val_image_names = val_list[:,0]
    val_labels = val_list[:,[1,2]].astype(np.float32)

    # 封装为 dataset
    train_set = MyDataSet(train_image_names,train_labels,input_transform)
    val_set = MyDataSet(val_image_names,val_labels,input_transform)
    #封装为 dateloader
    batchSize = cfg['batchSize']
    train_loader = DataLoader(dataset=train_set,batch_size=batchSize,shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=batchSize,shuffle=False)

    return train_loader , val_loader , val_image_names,train_image_names


class MyDataSet(Dataset):

    def __init__(self, x_img_name, y_label, transforms=None):
        self.x_img_name = x_img_name
        self.y_label = y_label
        self.transforms = transforms

    def __getitem__(self, item):
        X_img, Y_label = loadImgs(self.x_img_name[item], self.y_label[item].reshape(1,2))
        if self.transforms is not None:
            X_img = self.transforms(X_img)
        return X_img, Y_label.astype(np.float32)

    def __len__(self):
        return len(self.x_img_name)


if __name__ == '__main__':


    # random_list = random.sample(range(0,5),5)
    # tLD,vLD,_ = getDataload()
    # for i,data in enumerate(tLD):
    #     print(data)
    # randomlist = random.sample(range(0, 11), 11)
    # print(randomlist)
    # loadData('./data')
    # dataload()
    pass
















