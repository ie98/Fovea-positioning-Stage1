import cv2
import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
basePath = '../dataset/MESSIDOR'

imgName = '20051019_38557_0100_PP.tif'

img = cv2.imread(os.path.join(basePath,'images',imgName),-1)

print(img)

def getImg(imgName):
    img = cv2.imread(os.path.join(basePath,'images', imgName), -1)
    return img


labelFileName = 'data.xls'



def getLabels(labelFilesName):
    labels = pd.read_excel(os.path.join(basePath, 'label', labelFileName))
    return labels




def loadImgs(img_name,label):
    R = None

    downSample_W = 512
    downSample_H = 512

    img = cv2.imread(os.path.join(basePath,'images', img_name))
    if img.shape[0] == 1536:
        R = 109
    elif img.shape[0] == 1488:
        R = 103
    else:
        R = 68


    # 图像下采样
    img.astype(np.uint8)
    img = cv2.resize(img, dsize=(downSample_H, downSample_W), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('./{}.jpg'.format(img_name),img)
    return img,label.reshape(-1),R

class MyDataSet(Dataset):

    def __init__(self, x_img_name, y_label, transforms=None):
        self.x_img_name = x_img_name
        self.y_label = y_label
        self.transforms = transforms

    def __getitem__(self, item):
        X_img, Y_label,R = loadImgs(self.x_img_name[item], self.y_label[item].reshape(1,2))
        if self.transforms is not None:
            X_img = self.transforms(X_img)
        return X_img, Y_label.astype(np.float32),R

    def __len__(self):
        return len(self.x_img_name)



def getDataLoader():
    labelFileName = 'data.xls'
    labelDetail = np.array(getLabels(labelFileName))
    imgNames = labelDetail[:,0]
    labels = labelDetail[:, [2, 3]]
    random_list = random.sample(range(0, len(imgNames)), len(imgNames))
    np_list = np.array(random_list)
    imgNames = imgNames[np_list]
    labels = labels[np_list]
    test_img_names = imgNames
    test_label = labels.astype(np.float32)

    # 图片处理
    input_transform = transforms.Compose([
        # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
    ])


    test_set = MyDataSet(test_img_names,test_label,input_transform)
    test_dataloader = DataLoader(test_set,batch_size=1,shuffle=False)

    return test_dataloader,test_img_names



if __name__ == '__main__':
    train_loader,_ = getDataLoader()
    for i,data in enumerate(train_loader):
        print(data)