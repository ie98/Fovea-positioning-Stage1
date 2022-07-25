import numpy as np
import torch
import dataprocess.test_all_dataloader as DL
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
        return np.sqrt( temp[:,0] + temp[:,1] )


    # def getTrainData(self):
    #     can_use_imgs_dict, label , image_names = DL.loadLabel()
    #     imgs, new_labels = DL.loadImgs(can_use_imgs_dict, label)
    #     train_loader , val_loader , val_image_names = DL.dataload(imgs, new_labels,image_names)
    #     return train_loader, val_loader , val_image_names


    def getTemplate(self,templatePath):
        input_transform = transforms.Compose([
            # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
            transforms.ToPILImage(),
            transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
        ])
        template = cv2.imread(templatePath)  # 查询图像
        template = input_transform(template).unsqueeze(0)
        return template

    def predUp(self,pred,R):
        up_pred = np.zeros(shape=(1,2))
        if int(R) == 68:
            up_pred[0,0] = pred[0,0]*(960/512)
            up_pred[0,1] = pred[0,1]*(1440/512)
        elif int(R) == 103:
            up_pred[0, 0] = pred[0, 0] * (1488 / 512)
            up_pred[0, 1] = pred[0, 1] * (2240 / 512)
        else:
            up_pred[0, 0] = pred[0, 0] * (1536 / 512)
            up_pred[0, 1] = pred[0, 1] * (2304 / 512)
        return up_pred

    def train(self):


        device = 0
        model = mainNet()
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1],output_device=0)
        paramDict = torch.load('./modelParam/MESSIDOR/5/modelParam_358_5.981427205933465.pth')
        model.load_state_dict(paramDict['modelParam'])
        model.to(device)
        test_dataloader , test_imgNames = DL.getDataLoader()

        # 获取模板图片
        templatePath = './template_50.jpg'
        template = self.getTemplate(templatePath)

        R_list = []
        rate_R = []

        dist = 0.
        label_site = None
        pred_site = None
        curr_dist = None
        val_detail = None
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                R_list.append(int(data[2]))
                val_pred = model(data[0].cuda(device), data[0].detach().cuda(device),template.cuda(device))

                up_pred = self.predUp(val_pred.detach().cpu().numpy(),data[2])
                # batch_loss = sum([(x - y) ** 2 for x, y in zip(data[1].cuda(), val_pred)]) / len(train_pred)

                # 模型成绩测试
                dist = self.eucliDist(up_pred,data[1].detach().cpu().numpy())
                if curr_dist is None:
                    curr_dist = dist
                else:
                    curr_dist = np.concatenate((curr_dist,dist),axis=0)

                if label_site is None:
                    label_site = data[1].detach().cpu().numpy()
                else:
                    label_site = np.concatenate((label_site,data[1].detach().cpu().numpy()),axis=0)

                if pred_site is None:
                    pred_site = up_pred
                else:
                    pred_site = np.concatenate((pred_site,up_pred),axis=0)

                rate = dist/int(data[2])

                rate_R.append(float(rate))




                # s1, s2 = self.estimate(val_pred.cpu().clone().detach().numpy(), data[2].cpu().detach())
                # score1 = score1 + s1
                # # s2 = self.estimate(val_pred, data[2].cpu().detach())
                # score2 = score2 + s2

            # val_detail = np.concatenate(val_image_names, label_site, pred_site, curr_dist)



            dataframe = pd.DataFrame(
                {'image_name': test_imgNames.reshape(-1),
                 'label_site_x': label_site[:, 0].reshape(-1),
                 'label_site_y': label_site[:, 1].reshape(-1),
                 'pred_site_x': pred_site[:, 0].reshape(-1),
                 'pred_site_y': pred_site[:, 1].reshape(-1),
                 'curr_dist': curr_dist.reshape(-1),
                 'R':R_list,
                 'rate_R':rate_R})
            dataframe.to_csv('./test_in_all_img_5.csv', sep=',')









if __name__ == '__main__':
    trianer = trainer()
    trianer.train()
