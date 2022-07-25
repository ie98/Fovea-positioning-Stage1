import numpy as np
import torch
import dataprocess.crossVal_T_V_T as DL
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
from vat import VATLoss
from getDetail.computeRrate import computerRrate_val
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

saveFileName = 'crossval valset 10-2 &train in network & using crossVal & VAT & bs 4 & split strid 1 & label not int & newLossFunction & realR8rate'
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
        # 使用tensorBoard
        print('Strat Tensorboard with "tensorboard -- logdir=runs",view at http://locahost:6006/')
        tb_writer = SummaryWriter()

        device = device
        model = mainNet().cuda(device)
        # Dict = torch.load('/home/stu013/mapping/HBDW_v3/modelParam/MESSIDOR/train in network using crossVal not dataAug bs 4 mult split strid 1/modelParam_155_3.605979555089709.pth')
        # Dict = torch.load('/home/stu013/mapping/HBDW_v3/modelParam/MESSIDOR/train in network using crossVal not dataAug bs 4/modelParam_173_3.4735212930491275.pth',map_location={'cuda:0':'cuda:1'})

        #Dict = torch.load('/home/stu013/mapping/HBDW_v3/modelParam/MESSIDOR/train in network & using crossVal & not dataAug & bs 4 & split strid 1 & label not int & newLossFunction & realR8rate/modelParam_167_3.516147768833268.pth',map_location={'cuda:0':'cuda:1'})
        #paramDict = Dict['modelParam']
        #model.load_state_dict(paramDict)
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1],output_device=0)

        # vat损失
        vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)

        train_dataLoader, val_dateLoader, _, val_image_names, _ = DL.getDataLoader(4)
        # np.savetxt('vn.txt',val_image_names)
        num_epoch = 300
        loss = nn.MSELoss(reduction='mean')
        nn.CrossEntropyLoss()
        learning_rate = 0.001
        lrf = 0.1
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Scheduler 学习率下降曲线
        lf = lambda x: ((1 + math.cos((x) * math.pi / num_epoch)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # 获取模板图片
        templatePath1 = './template_50.jpg'

        template1 = self.getTemplate(templatePath1)

        computer = computerRrate_val()



        # 保存每个iteration的loss和accuracy，以便后续画图
        plt_train_loss = []
        plt_val_loss = []
        # plt_train_acc = []
        # plt_val_acc = []

        # 用测试集训练模型model(),用验证集作为测试集来验证
        curr_lr = learning_rate
        best_dist = 99999999
        best_loss = 999999
        best_model = {}

        for epoch in range(num_epoch):
            epoch_start_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0
            dist = 0.
            label_site = None
            pred_site = None
            curr_dist = None
            val_detail = None
            R_8 = 0
            model.train()  # 确保 model 是在 训练 model (开启 Dropout 等...)
            #if epoch % 2 == 0:
             #   train_LD = train_dataLoader
            #    val_LD = val_dateLoader
            #else:
            #    train_LD = val_dateLoader
            #    val_LD = train_dataLoader
            train_LD = train_dataLoader
            val_LD = val_dateLoader
            for i, data in tqdm(enumerate(train_LD)):

                # print(data[0].shape)
                # x.append(data[0].cuda())
                # x.append(data[1].cuda())
                optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零
                if epoch > 3:
                    lds = vat_loss(model, data[0].cuda(device), data[0].detach().cuda(device),template1.cuda(device))

                main_Pred = model(data[0].cuda(device), data[0].detach().cuda(device),template1.cuda(device))  # 利用 model 得到预测的概率分布，这边实际上是调用模型的 forward 函数
                main_loss = self.lossFunction(main_Pred.cuda(device), data[1].cuda(device))
                if epoch > 3:
                    batch_loss = main_loss + 0.2*(epoch/num_epoch) * lds # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
                else:
                    batch_loss = main_loss
                batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
                optimizer.step()  # 以 optimizer 用 gradient 更新参数

                # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                train_loss += batch_loss.item()

            scheduler.step()

            # 验证集val
            model.eval()
            with torch.no_grad():

                for i, data in tqdm(enumerate(val_LD)):
                    val_pred = model(data[0].cuda(device), data[0].detach().cuda(device),template1.cuda(device))

                    # batch_loss = sum([(x - y) ** 2 for x, y in zip(data[1].cuda(), val_pred)]) / len(train_pred)
                    batch_loss = self.lossFunction(val_pred.cuda(device), data[1].cuda(device))
                    # val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())


                    val_loss += batch_loss.item()
                    # 模型成绩测试
                    dist_list = self.eucliDist(val_pred.detach().cpu().numpy(),data[1].detach().cpu().numpy())
                    R_8_list = dist_list[dist_list < 3.557]
                    R_8 = R_8 + len(R_8_list)
                    if curr_dist is None:
                        curr_dist = dist_list
                    else:
                        curr_dist = np.concatenate((curr_dist,dist_list),axis=0)

                    if label_site is None:
                        label_site = data[1].detach().cpu().numpy()
                    else:
                        label_site = np.concatenate((label_site,data[1].detach().cpu().numpy()),axis=0)

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

                dist = dist / 114 # 验证图片有 568 张
                R_8_per = R_8 / 114
                print('avg_dist = {}'.format(dist))
                logging.info('avg_dist = {}'.format(dist))

                print('R_8_per = {}'.format(R_8_per))
                logging.info('R_8_per = {}'.format(R_8_per))
                # 保存用于画图
                # plt_train_acc.append(train_acc / 1723)

                count = computer.computer(pred_site, os.path.join('outputFile/MESSIDOR', saveFileName), epoch, val_image_names.reshape(-1))

                real_r8_rate = count / 114
                print('real_r8_reate = {}'.format(real_r8_rate))
                logging.info('real_r8_rate = {}'.format(real_r8_rate))

                plt_train_loss.append(train_loss / train_LD.__len__())
                # plt_val_acc.append(val_acc / 431)
                plt_val_loss.append(val_loss / val_LD.__len__())

                dataframe = pd.DataFrame(
                    {'image_name': val_image_names.reshape(-1), 'label_site_x': label_site[:, 0].reshape(-1),
                     'label_site_y': label_site[:, 1].reshape(-1),
                     'pred_site_x': pred_site[:, 0].reshape(-1),
                     'pred_site_y': pred_site[:, 1].reshape(-1),
                     'curr_dist': curr_dist.reshape(-1)})
                if os.path.exists('./outputFile/MESSIDOR/{}'.format(saveFileName)) is False:
                    os.makedirs('./outputFile/MESSIDOR/{}'.format(saveFileName))
                dataframe.to_csv('./outputFile/MESSIDOR/{}/val_detail_{}.csv'.format(saveFileName,epoch), sep=',')
                # 将结果 print 出來

                print('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f | Val  loss: %3.6f' % \
                      (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                       plt_train_loss[-1], plt_val_loss[-1]))

                logging.info('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f | Val  loss: %3.6f' % \
                      (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                       plt_train_loss[-1], plt_val_loss[-1]))

                tags = ["train_loss","val_loss", "val_dist", "learning_rate"]
                tb_writer.add_scalar(tags[0], plt_train_loss[-1], epoch)
                tb_writer.add_scalar(tags[1],plt_val_loss[-1],epoch)
                tb_writer.add_scalar(tags[2], dist, epoch)
                tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

                if epoch >= 120 :
                    if dist < best_dist:
                        if os.path.exists('./modelParam/MESSIDOR/{}'.format(saveFileName)) is False:
                            os.makedirs('./modelParam/MESSIDOR/{}'.format(saveFileName))

                        best_dist = dist
                        best_model['dist'] = dist
                        best_model['modelParam'] = model.state_dict()
                        torch.save(best_model,'./modelParam/MESSIDOR/{}/modelParam_{}_{}.pth'.format(saveFileName,epoch,dist))



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
    trianer.train(0)

