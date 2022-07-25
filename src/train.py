
import sys
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from multi_train_utils.distributed_utils import reduce_value, is_main_process


def train_one_epoch(model, optimizer, data_loader, device, epoch,template):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device),images.detach().to(device),template.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] train mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()

def eucliDist(self,pre,label):
    temp = np.square( pre - label )
    return np.sqrt( temp[:,0] + temp[:,1] )

@torch.no_grad()
def evaluate(model, data_loader, device,template,epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()


    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    mean_loss = torch.zeros(1).to(device)
    mean_dist = torch.zeros(1).to(device)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device),images.detach().to(device),template.to(device))
        loss = loss_function(pred, labels.to(device))
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 模型成绩测试
        last_dist = np.sum(eucliDist(pred.detach().cpu().numpy(), labels.detach().cpu().numpy()))
        last_dist = reduce_value(last_dist,average=True)
        mean_dist = (mean_dist * step + last_dist.detach()) / (step + 1)  # update mean dist
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] val mean loss {} , val mean dist {}".format(epoch, round(mean_loss.item(), 3),round(mean_dist.item(),3))

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item(),mean_dist.item()
