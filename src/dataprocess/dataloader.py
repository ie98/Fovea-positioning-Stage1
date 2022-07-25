import os.path
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

print(os.path.abspath(''))
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
    for i in tqdm(range(image_names.shape[0])):
    # for i in tqdm(range(100)):
        if label[i, 0] <= 50 and label[i, 1] <= 50:
            # list.append(image_names[i])
            continue
        if label[i, 1] >= int(cv2.imread(os.path.join(root_path, str(image_names[i]))).shape[1] * 0.7):
            # list_right.append(image_names[i])
            continue
        can_use_imgs_dict[image_names[i]] = i
        can_use_imgname_list.append(image_names[i])
    return can_use_imgs_dict,label,np.array(can_use_imgname_list)


def loadImgs(can_use_imgs_dict:dict,label):
    downSample_W = cfg['downSample_W']
    downSample_H = cfg['downSample_H']
    imgs = np.ndarray(shape=[len(can_use_imgs_dict), downSample_W, downSample_H, 3])
    new_labels = np.ndarray(shape=[len(can_use_imgs_dict), 2])
    i = 0
    for img_name in tqdm(can_use_imgs_dict.keys()):

        img = cv2.imread(os.path.join(root_path, img_name))

        # 计算图像下采样后，标签的位置
        rate_x = 512 / img.shape[1]
        rate_y = 512 / img.shape[0]
        new_label_x = int( label[can_use_imgs_dict[img_name],1] * rate_x)
        new_label_y = int( label[can_use_imgs_dict[img_name],0] * rate_y)
        new_labels[i,0] = new_label_y
        new_labels[i,1] = new_label_x

        # 图像下采样
        img.astype(np.uint8)
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        imgs[i,:] = img
        i = i + 1



    return imgs,new_labels

def dataload(imgs,labels,image_names):
    # 图片处理
    input_transform = transforms.Compose([
        # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
    ])
    # 训练集和验证集的比例
    rate = cfg['dataRate']
    # 打乱数据集和标签
    random_list = random.sample(range(0, imgs.shape[0]), imgs.shape[0])
    np_list = np.array(random_list)
    val_image_names = image_names[np_list[int(imgs.shape[0]*rate):]]
    # np.savetxt('val_image_names.txt', val_image_names, delimiter=',')

    imgs = imgs[random_list,:].astype(np.uint8)
    labels = labels[random_list,:].astype(np.float32)
    # 切分训练集和验证集
    train_imgs = imgs[0:int(imgs.shape[0]*rate),:]
    train_labels = labels[0:int(imgs.shape[0]*rate),:]
    val_imgs = imgs[int(imgs.shape[0]*rate):,]
    val_labels = labels[int(imgs.shape[0]*rate):,]
    # 封装为 dataset
    train_set = MyDataSet(train_imgs,train_labels,input_transform)
    val_set = MyDataSet(val_imgs,val_labels,input_transform)
    #封装为 dateloader
    batchSize = cfg['batchSize']
    train_loader = DataLoader(dataset=train_set,batch_size=batchSize,shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=batchSize,shuffle=False)

    return train_loader , val_loader , val_image_names


def getDataLoader():
    can_use_imgs_dict, label,image_names = loadLabel()
    imgs, new_labels = loadImgs(can_use_imgs_dict, label)
    train_loader, val_loader , val_image_names = dataload(imgs, new_labels,image_names)
    return train_loader, val_loader , val_image_names


class MyDataSet(Dataset):

    def __init__(self, x_img, y_label, transforms=None):
        self.x_img = x_img
        self.y_label = y_label
        self.transforms = transforms

    def __getitem__(self, item):
        if self.transforms is not None:
            X_img = self.transforms(self.x_img[item])
        else:
            X_img = self.x_img[item]
        Y_label = self.y_label[item]

        return X_img, Y_label

    def __len__(self):
        return len(self.x_img)


if __name__ == '__main__':


    random_list = random.sample(range(0,5),5)
    getDataLoader()
















