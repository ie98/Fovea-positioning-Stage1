import cv2
import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
basePath = '../dataset/IDRID'

imgName = '20051019_38557_0100_PP.tif'

img = cv2.imread(os.path.join(basePath,'images',imgName),-1)

print(img)

def getImg(imgName):
    img = cv2.imread(os.path.join(basePath,'images', imgName), -1)
    return img




def getLabels(labelFilesName):
    labels = pd.read_csv(os.path.join(basePath, '2. Groundtruths/2. Fovea Center Location', labelFilesName))
    labels = labels.dropna(how='all')
    return labels




def loadImgs(img_name,label):
    downSample_W = 512
    downSample_H = 512
    new_labels = np.ndarray(shape=[1, 2])
    img = cv2.imread(os.path.join(basePath,'1. Original Images', img_name+'.jpg'))

    # 计算图像下采样后，标签的位置
    rate_x = 512 / img.shape[1]
    rate_y = 512 / img.shape[0]
    new_label_x =  label[0,0] * rate_x
    new_label_y =  label[0,1] * rate_y
    new_labels[0,0] = new_label_x
    new_labels[0,1] = new_label_y

    # 图像下采样
    img.astype(np.uint8)
    img = cv2.resize(img, dsize=(downSample_H, downSample_W), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('./{}.jpg'.format(img_name),img)
    return img,new_labels.reshape(-1)

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




def getDataLoader(batchSize):
    # 训练集和测试集都拿来测试
    labelFileName_test = 'IDRiD_Fovea_Center_Testing Set_Markups.csv'
    labelFileName_train = 'IDRiD_Fovea_Center_Training Set_Markups.csv'

    labelDetail_train = np.array(getLabels(labelFileName_train))
    labelDetail_test = np.array(getLabels(labelFileName_test))

    imgNames = np.concatenate(('a. Training Set/'+labelDetail_train[:,0],'b. Testing Set/'+labelDetail_test[:,0]),axis=0)
    labels = np.concatenate((labelDetail_train[:, [1, 2]],labelDetail_test[:, [1, 2]]),axis=0)

    # random_list = random.sample(range(0, len(imgNames)), len(imgNames))
    # np_list = np.array(random_list)
    # imgNames = imgNames[np_list]
    # labels = labels[np_list]
    # rate = 0.5
    # train_imgNames = imgNames[:int(len(imgNames) * rate)]
    # train_labels = labels[:int(len(labels) * rate), :].astype(np.float32)
    # val_imgName = imgNames[int(len(imgNames) * rate):]
    # val_labels = labels[int(len(labels) * rate):, :].astype(np.float32)
    #subset_1 = np.array(pd.read_csv('./subSet/subSet_1.csv'))[:,1:]
    #subset_2 = np.array(pd.read_csv('./subSet/subSet_2.csv'))[:,1:]
    #random_list = random.sample(range(0, len(subset_1)), len(subset_1))
    #rand_arry = np.array(random_list)
    #subset_1 = subset_1[rand_arry]
    #train_imgNames = subset_1[:,0]
    #train_labels = subset_1[:,[1,2]].astype(np.float32)
    #val_imgName = subset_2[:,0]
    #val_labels = subset_2[:,[1,2]].astype(np.float32)
    # 图片处理
    input_transform = transforms.Compose([
        # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
        #transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
    ])


    test_set = MyDataSet(imgNames,labels,input_transform)
    test_loader = DataLoader(test_set,batch_size=batchSize,shuffle=False,num_workers=2)
    return test_loader,imgNames,labels



if __name__ == '__main__':
    train_loader,_ = getDataLoader(4)
    for i,data in enumerate(train_loader):
        print(data)