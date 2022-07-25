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

def loadLabel():

    excel = pd.read_excel('../dataset/Train/Fovea_Location_train.xlsx')
    excel = np.array(excel)
    label = excel[:, [2, 1]] # 标签中是以 【列，行】 表示 ， [2,1] 转为 【行 ， 列】
    image_names = excel[:, 0]
    # list = []
    # list_right = []
    can_use_imgs_dict = {}
    can_use_imgname_list=[]
    can_use_labels = np.zeros(shape=[1,2])
    for i in tqdm(range(image_names.shape[0])):
    # for i in tqdm(range(50)):
        if label[i, 0] <= 50 and label[i, 1] <= 50:
            # list.append(image_names[i])
            continue
        if label[i, 1] >= int(cv2.imread(os.path.join(root_path, str(image_names[i]))).shape[1] * 0.7):
            # list_right.append(image_names[i])
            continue
        can_use_imgs_dict[image_names[i]] = i
        can_use_imgname_list.append(image_names[i])
        if i == 0:
            can_use_labels[0,:] = label[i,:]
        else:
            can_use_labels = np.concatenate((can_use_labels,label[i,:].reshape(1,2)),0)
    return can_use_imgs_dict,can_use_labels,np.array(can_use_imgname_list)


def loadImgs(img_name,label):
    downSample_W = cfg['downSample_W']
    downSample_H = cfg['downSample_H']
    new_labels = np.ndarray(shape=[1, 2])
    img = cv2.imread(os.path.join(root_path, img_name))

    # 计算图像下采样后，标签的位置
    rate_x = 512 / img.shape[1]
    rate_y = 512 / img.shape[0]
    new_label_x = int( label[0,1] * rate_x)
    new_label_y = int( label[0,0] * rate_y)
    new_labels[0,0] = new_label_y
    new_labels[0,1] = new_label_x

    # 图像下采样
    img.astype(np.uint8)
    img = cv2.resize(img, dsize=(downSample_H, downSample_W), interpolation=cv2.INTER_LINEAR)

    return img,new_labels.reshape(-1)

def dataload(labels,image_names):
    # 图片处理
    input_transform = transforms.Compose([
        # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
    ])

    test_set = MyDataSet(image_names,labels,input_transform)
    #封装为 dateloader
    batchSize = cfg['batchSize']
    test_loader = DataLoader(dataset=test_set,batch_size=batchSize,shuffle=False)

    return test_loader


def getDataLoader():
    can_use_imgs_dict, label,image_names = loadLabel()
    # imgs, new_labels = loadImgs(can_use_imgs_dict, label)
    train_loader, val_loader , val_image_names = dataload(label,image_names)
    return train_loader, val_loader , val_image_names


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


    random_list = random.sample(range(0,5),5)
    tLD,vLD,_ = getDataLoader()
    for i,data in enumerate(tLD):
        print(data)
















