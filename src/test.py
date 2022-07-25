import numpy as np
import torch
import dataprocess.dataLoader_3 as DL
from network.mainNet.mainNet import mainNet
import torch.nn as nn
import time
import cv2
from torchvision import transforms
import pandas as pd
import os
import math
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
class trainer():

    def __init__(self,arg_file=None):
        pass

    def eucliDist(self,pre,label):
        temp = np.square( pre - label )
        return np.sqrt( temp[0,0] + temp[0,1] )


    # def getTrainData(self):
    #     can_use_imgs_dict, label , image_names = DL.loadLabel()
    #     imgs, new_labels = DL.loadImgs(can_use_imgs_dict, label)
    #     train_loader , val_loader , val_image_names = DL.dataload(imgs, new_labels,image_names)
    #     return train_loader, val_loader , val_image_names


    def getTemplate(self,templatePath,H,W):
        input_transform = transforms.Compose([
            # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
            transforms.ToPILImage(),
            transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
        ])
        template = cv2.imread(templatePath)  # 查询图像
        template = input_transform(cv2.resize(template, dsize=(H, W))).unsqueeze(0) # 默认 54
        return template

    def getImageNameAndLabel(self):
        testImgDetial = np.array(pd.read_csv('outputFile/MESSIDOR/2/val_detail_145.csv'))
        all_labels = np.array(pd.read_excel('../dataset/MESSIDOR/label/data.xls'))
        test_imgNames = testImgDetial[:,1 ]
        labels = None
        for i in range(len(test_imgNames)):
            flag = 1
            currImgName = test_imgNames[i]
            for j in range(len(all_labels)):
                if currImgName == all_labels[j,0]:
                    if labels is None:
                        labels = all_labels[j,:].reshape(1,-1)
                    else:
                        labels = np.concatenate((labels,all_labels[j,:].reshape(1,-1)),axis=0)
                    flag = 0
                    break
            if flag == 1:
                print('{} is error!! '.format(currImgName))

        return test_imgNames,labels[:,[2,3]]
    def getImage(self,imageName):
        basePath = '../dataset/MESSIDOR/images'
        img = cv2.imread(os.path.join(basePath,imageName))
        return img

    def train(self):
        input_transform = transforms.Compose([
            # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
            transforms.ToPILImage(),
            transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
        ])
        test_imgNames,labels = self.getImageNameAndLabel()

        # 获取模板图片
        templatePath = './template.jpg'


        device = 0
        model = mainNet().cuda(device)
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1],output_device=0)
        paramDict = torch.load('./modelParam/MESSIDOR/2/modelParam_145_4.19096451056631_16.565404949517085.pth')
        model.load_state_dict(paramDict['modelParam'])
        R = []





        # 验证集val
        model.eval()
        with torch.no_grad():
            for i in range(len(test_imgNames)):
                image = self.getImage(test_imgNames[i])
                # image = cv2.resize(image, dsize=(768, 512), interpolation=cv2.INTER_LINEAR)
                # H = W = int((image.shape[0]/512)*54)
                H=W=54
                template = self.getTemplate(templatePath,H,W)
                if image.shape[0] == 1536:
                    R.append(109)
                elif image.shape[0] == 1488:
                    R.append(103)
                else:
                    R.append(68)
                label = labels[i]
                image = input_transform(image)
                image = image.unsqueeze(0)
                val_pred = model(image.cuda(device), image.detach().cuda(device),template.cuda(device))
                # batch_loss = sum([(x - y) ** 2 for x, y in zip(data[1].cuda(), val_pred)]) / len(train_pred)
                # 模型成绩测试
                dist_list = self.eucliDist(val_pred.detach().cpu().numpy(),label.reshape(1,-1))
                if curr_dist is None:
                    curr_dist = dist_list
                else:
                    curr_dist = np.concatenate((curr_dist,dist_list),axis=0)

                if label_site is None:
                    label_site = label.reshape(1,-1)
                else:
                    label_site = np.concatenate((label_site,label.reshape(1,-1)),axis=0)

                if pred_site is None:
                    pred_site = val_pred.detach().cpu().numpy()
                else:
                    pred_site = np.concatenate((pred_site,val_pred.detach().cpu().numpy()),axis=0)



                dist = dist + np.sum(dist_list)

                # s1, s2 = self.estimate(val_pred.cpu().clone().detach().numpy(), data[2].cpu().detach())
                # score1 = score1 + s1
                # # s2 = self.estimate(val_pred, data[2].cpu().detach())
                # score2 = score2 + s2

            # val_detail = np.concatenate(val_image_names, label_site, pred_site, curr_dist)

            dist = dist / 228 # 验证图片有 228 张
            print('avg_dist = {}'.format(dist))

            dataframe = pd.DataFrame(
                {'image_name': test_imgNames.reshape(-1), 'label_site_x': label_site[:, 0].reshape(-1),
                 'label_site_y': label_site[:, 1].reshape(-1),
                 'pred_site_x': pred_site[:, 0].reshape(-1),
                 'pred_site_y': pred_site[:, 1].reshape(-1),
                 'curr_dist': curr_dist.reshape(-1),
                 'R': R})
            dataframe.to_csv('./outputFile/MESSIDOR/2/val_detail_allSize.csv', sep=',')





if __name__ == '__main__':
    trianer = trainer()
    trianer.train()
