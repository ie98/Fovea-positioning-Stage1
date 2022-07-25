import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from src.network_2_2.SPP.sppmodule import PSPModule
from src.network_2_2.v1.fm1.FeatureMatch_1_1 import FeatureMatch


class mainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18_0_5 = nn.Sequential(*list(self.resnet18.children())[:6])
        # self.resnet18_6 = nn.Sequential(*list(self.resnet18.children())[6])
        # self.resnet18_6_8_fm1 = nn.Sequential(*list(resnet18(pretrained=True).children())[6:9])
        # self.resnet18_6_8_fm2 = nn.Sequential(*list(resnet18(pretrained=True).children())[6:9])
        self.resnet18_7 = nn.Sequential(*list(self.resnet18.children())[7])
        self.model2 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.spp = PSPModule(512)
        self.featureMatch = FeatureMatch()
        self.gap_x = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.gap_fm1 = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.gap_fm2 = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.gap_7 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.pre = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

        self.mult_1_1 = nn.Sequential(
            nn.Conv2d(512,128,kernel_size=1),
            nn.AdaptiveAvgPool2d(output_size=(8, 8)), ### 此处有问题，输出的尺寸不一定是 16 * 16
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.mult_1_2 = nn.Sequential(
            nn.Conv2d(512,128,kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)

        )
        self.mult_2_1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.AdaptiveAvgPool2d(output_size=(16, 16)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.mult_2_2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.attention_1_1 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=1)
        )
        self.attention_1_2 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1)
        )
        self.attention_1_3 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1)
        )

        self.attention_2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.attention_2_2 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1)
        )
        self.attention_2_3 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1)
        )
        self.avgp1 = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.avgp2 = nn.AdaptiveAvgPool2d(output_size=(3, 3))

        self.fc1 = nn.Sequential(

            nn.Linear(3*3*128,2)
        )
        self.fc2 = nn.Sequential(

            nn.Linear(3 * 3 * 128, 2)
        )

        self.fm_1_fc = nn.Linear(512,2)
        self.fm_2_fc = nn.Linear(512,2)

    def mult(self,x):
        x1 = self.model2(x)  # 512 32 32
        x1_1 = self.mult_1_1(x1)  # 128 16 16
        x1_2 = self.mult_1_2(x1)  # 128 32 32
        x2 = self.model3(x1)  # 512 16 16
        x2_1 = self.mult_2_1(x2) # 128 32 32
        x2_2 = self.mult_2_2(x2) # 128 16 16
        x_1_add = x1_1 + x2_2  # 128 16 16
        x_2_add = x2_1 + x1_2  # 128 32 32
        return x_1_add,x_2_add

    def selfAttention_1(self,x_1_add):
        x_a1_1 = self.attention_1_1(x_1_add)
        x_a1_1 = x_a1_1.view(x_a1_1.size(0), x_a1_1.size(1), -1)
        x_a1_2 = self.attention_1_2(x_1_add)
        x_a1_2 = x_a1_2.view(x_a1_2.size(0), x_a1_2.size(1), -1)
        x_a1_3 = self.attention_1_3(x_1_add)
        x_a1_3 = x_a1_3.view(x_a1_3.size(0), x_a1_3.size(1), -1)

        x_a1_2 = torch.transpose(x_a1_2,1,2)
        x_m1_1 = torch.matmul(x_a1_2, x_a1_3)
        x_m1_1 = F.softmax(x_m1_1, dim=2)
        # sum1 = torch.sum(x_m1_1[0,1,:])
        # sum50 = torch.sum(x_m1_1[0,50,:])
        x_m1_1 = torch.transpose(x_m1_1,1,2)
        # sum1 = torch.sum(x_m1_1[0, :, 1])
        # sum50 = torch.sum(x_m1_1,dim=1)
        x_Aend_1 = torch.matmul(x_a1_1, x_m1_1)
        return x_Aend_1

    def selfAttention_2(self,x_2_add):
        # 第二个自注意力
        x_a2_1 = self.attention_1_1(x_2_add)
        x_a2_1 = x_a2_1.view(x_a2_1.size(0), x_a2_1.size(1), -1)
        x_a2_2 = self.attention_1_2(x_2_add)
        x_a2_2 = x_a2_2.view(x_a2_2.size(0), x_a2_2.size(1), -1)
        x_a2_3 = self.attention_1_3(x_2_add)
        x_a2_3 = x_a2_3.view(x_a2_3.size(0), x_a2_3.size(1), -1)
        x_a2_2 = torch.transpose(x_a2_2,1,2)
        x_m2_1 = torch.matmul(x_a2_2, x_a2_3)
        x_m2_1 = F.softmax(x_m2_1, dim=2)
        # sum1 = torch.sum(x_m2_1[0, 1, :])
        # sum50 = torch.sum(x_m2_1[0, 50, :])
        x_m2_1 = torch.transpose(x_m2_1,1,2)
        x_Aend_2 = torch.matmul(x_a2_1, x_m2_1)
        return x_Aend_2

    def forward(self,x_all,x_crop,t1,t2=None):

        x = self.resnet18_0_5(x_all) # 128 64 64
        ###### 辅助路线
        fm1 = self.featureMatch(x_crop,t1) # 128 113 113  /  128 121 121  # 考虑对fm进行损失计算
        x_T = self.gap_x(x)  # 128 64 64
        fm1 = self.gap_fm1(fm1) # 128 64 64
        # fm2 = self.gap_fm2(fm2) # 128 64 64
        #fm1_M_x = fm1*x_T #128 64 64
        # fm2_M_x = fm2*x_T #128 64 64
        # fm1_M_x = self.resnet18_6_8_fm1(fm1_M_x)
        # fm2_M_x = self.resnet18_6_8_fm2(fm2_M_x)
        ###### 主路线
        fm = torch.cat((fm1,x_T),dim=1) # 256 64 64
        # x = self.resnet18_6(fm1_M_x)  # 256 32 32
        # 多尺度融合
        # x_1_add , x_2_add = self.mult(x) # 128 16 16 / 128 32 32
        # # 自注意力机制
        # x_Aend_1 = self.selfAttention_1(x_1_add) #128 256
        # x_Aend_1 = self.avgp1(x_Aend_1.view(x_Aend_1.size(0),x_Aend_1.size(1),8,8))
        # x_Aend_2 = self.selfAttention_2(x_2_add) # 128 1024
        # x_Aend_2 = self.avgp2(x_Aend_2.view(x_Aend_2.size(0),x_Aend_2.size(1),16,16))
        # # 全连接层
        # res1 = self.fc1(x_Aend_1.view(x_Aend_1.size(0),-1))
        # res2 = self.fc2(x_Aend_2.view(x_Aend_2.size(0),-1))
        # mainPred = (res1+res2)/2
        # fm1Pred = self.fm_1_fc(fm1_M_x.view(fm1_M_x.size(0),-1))
        # fm2Pred = self.fm_2_fc(fm2_M_x.view(fm2_M_x.size(0),-1))
        # return mainPred
        x = self.resnet18_7(fm)  # 256 32 32

        x = self.spp(x)

        x = self.gap_7(x)

        x = x.view(x.size(0), -1)

        return self.pre(x)






#
#
# if __name__ == '__main__':
#     x = torch.zeros([1,3,512,512])
#     net = mainNet()
#     res = net(x,x,x)
#
#     print('strat'.center(40,'*'))
#     summary(net , [x,x,x])
#     print('end'.center(40,'*'))
#
