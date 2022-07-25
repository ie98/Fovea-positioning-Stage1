import numpy as np
import torch
import dataprocess.crossval_trainInSubset1_testInSubset2 as DL
# from network.mainNet.mainNet import mainNet
from network.mainNet.mainNet import mainNet
import torch.nn as nn
import time
import cv2
from torchvision import transforms
import pandas as pd
import os
import math
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
from getDetail.computeRrate import computerRrate_test
from torchvision.models import resnet18
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from thop import profile
saveFileName = 'end test in subset2'
epoch_num = 265
class trainer():

    def __init__(self,arg_file=None):
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        if os.path.exists('./outputFile/MESSIDOR/{}'.format(saveFileName)) is False:
            os.makedirs('./outputFile/MESSIDOR/{}'.format(saveFileName))
        if os.path.exists('./outputFile/MESSIDOR/{}/my.log'.format(saveFileName)) is False:
            file = open('./outputFile/MESSIDOR/{}/my.log'.format(saveFileName), 'w')
            file.close()
        logging.basicConfig(filename='./outputFile/MESSIDOR/{}/my.log'.format(saveFileName), level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


    def eucliDist(self,pre,label):
        temp = np.square( pre - label )
        return np.sqrt( temp[:,0] + temp[:,1] )


    # def getTrainData(self):
    #     can_use_imgs_dict, label , image_names = DL.loadLabel()
    #     imgs, new_labels = DL.loadImgs(can_use_imgs_dict, label)
    #     train_loader , val_loader , val_image_names = DL.dataload(imgs, new_labels,image_names)
    #     return train_loader, val_loader , val_image_names

    def lossFunction(self,pred,label):
        loss = torch.mean(1.5*(label[:,0]-pred[:,0])**2+(label[:,1]-pred[:,1])**2)
        return loss

    def getTemplate(self,templatePath):
        input_transform = transforms.Compose([
            # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
            transforms.ToPILImage(),
            transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
        ])
        template = cv2.imread(templatePath)  # 查询图像
        template = template.astype(np.uint8)
        template = input_transform(template).unsqueeze(0)
        return template


    def train(self,device):

        detail = 'pre_subset2'

        # 使用tensorBoard
        print('Strat Tensorboard with "tensorboard -- logdir=runs",view at http://locahost:6006/')
        tb_writer = SummaryWriter()
        device = device
        model = resnet18(pretrained=False)
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 2)
        )
        model = model.cuda(device)
        Dict = torch.load('./End_use_model_Param/train_in_subset1.pth',map_location={'cuda:1':'cuda:0'})
        paramDict = Dict['modelParam']
        model.load_state_dict(paramDict)
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1],output_device=0)
        batchSize = 1
        train_loader,val_loader,test_loader,val_imgName,test_imgNames = DL.getDataLoader(batchSize)
        # np.savetxt('vn.txt',val_image_names)


        logging.info('batchsize is {}'.format(batchSize))

        computer = computerRrate_test(2)

        # 获取模板图片
        templatePath1 = './template_50.jpg'
        template1 = self.getTemplate(templatePath1)


        # 用测试集训练模型model(),用验证集作为测试集来验证

        dist = 0.
        label_site = None
        pred_site = None
        curr_dist = None
        R_8 = 0
        allTime = 0.

        # 验证集val
        model.eval()
        with torch.no_grad():

            for i, data in tqdm(enumerate(test_loader)):
                flops , params = profile(model,inputs = (data[0].cuda(device),))
                stratTime = time.time()
                val_pred = model(data[0].cuda(device))

                print('use time = {}'.format(time.time()-stratTime))
                allTime = allTime + time.time() - stratTime
            print('avg time = {}'.format(allTime/568))

            #     # 模型成绩测试
            #     dist_list = self.eucliDist(val_pred.detach().cpu().numpy(),data[1].detach().cpu().numpy())
            #     R_8_list = dist_list[dist_list < 3.557]
            #     R_8 = R_8 + len(R_8_list)
            #     if curr_dist is None:
            #         curr_dist = dist_list
            #     else:
            #         curr_dist = np.concatenate((curr_dist,dist_list),axis=0)
            #
            #     if label_site is None:
            #         label_site = data[1].detach().cpu().numpy()
            #     else:
            #         label_site = np.concatenate((label_site,data[1].detach().cpu().numpy()),axis=0)
            #
            #     if pred_site is None:
            #         pred_site = val_pred.detach().cpu().numpy()
            #     else:
            #         pred_site = np.concatenate((pred_site,val_pred.detach().cpu().numpy()),axis=0)
            #
            #     dist = dist + np.sum(dist_list)
            #
            #
            # dist = dist / 568 # 验证图片有 568 张
            # R_8_per = R_8 / 568
            # print('avg_dist = {}'.format(dist))
            # logging.info('avg_dist = {}'.format(dist))
            #
            # print('R_8_per = {}'.format(R_8_per))
            # logging.info('R_8_per = {}'.format(R_8_per))
            #
            # if os.path.exists('./testInMessidor/test_in_subset2') is False:
            #     os.makedirs('./testInMessidor/test_in_subset2')
            # count = computer.computer(pred_site,os.path.join('./testInMessidor/test_in_subset2',saveFileName),epoch_num,test_imgNames.reshape(-1))
            #
            # real_r8_rate = count/568
            # print('real_r8_reate = {}'.format(real_r8_rate))
            # logging.info('real_r8_rate = {}'.format(real_r8_rate))
            #
            #
            # dataframe = pd.DataFrame(
            #     {'image_name': test_imgNames.reshape(-1), 'label_site_x': label_site[:, 0].reshape(-1),
            #      'label_site_y': label_site[:, 1].reshape(-1),
            #      'pred_site_x': pred_site[:, 0].reshape(-1),
            #      'pred_site_y': pred_site[:, 1].reshape(-1),
            #      'curr_dist': curr_dist.reshape(-1)})
            # if os.path.exists('./testInMessidor/test_in_subset2/{}'.format(saveFileName)) is False:
            #     os.makedirs('./testInMessidor/test_in_subset2/{}'.format(saveFileName))
            # dataframe.to_csv('./testInMessidor/test_in_subset2/{}/test_{}.csv'.format(saveFileName,detail), sep=',')
            #
            #
            #



import pynvml


# 字节数转GB
def bytes_to_gb(sizes):
    sizes = round(sizes / (1024 ** 3), 2)
    return f"{sizes} GB"


def gpu(num):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(num)  # 显卡句柄（只有一张显卡）
    gpu_name = pynvml.nvmlDeviceGetName(handle)  # 显卡名称
    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 显卡内存信息
    data = dict(
        gpu_name=gpu_name.decode("utf-8"),
        gpu_memory_total=bytes_to_gb(gpu_memory.total),
        gpu_memory_used=bytes_to_gb(gpu_memory.used),
        gpu_memory_free=bytes_to_gb(gpu_memory.free),
    )
    return data


if __name__ == '__main__':

    device = 0
    var = 1
    i = 0
    # while var == 1:
    #     time.sleep(1)
    #     gpu_0 = gpu(0)
    #     gpu_1 = gpu(1)
    #
    #     free = []
    #     free.append(float(gpu_0['gpu_memory_free'].split(' ')[0]))
    #     free.append(float(gpu_1['gpu_memory_free'].split(' ')[0]))
    #     if free[0] > 6:
    #         device = 0
    #         break
    #     if free[1] > 6:
    #         device = 1
    #         break
    #
    #     if i%60==0:
    #         print(i)
    #     i = i+1

    trianer = trainer()
    trianer.train(device)

