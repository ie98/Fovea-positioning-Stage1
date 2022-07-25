from torch import nn
from torchvision.models import resnet18
from src.network_BWG.SPP.sppmodule import PSPModule
from src.network_BWG.v1.fm1.FeatureMatch_1_1 import FeatureMatch


class mainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.resnet18.children())[:6])

        self.spp_1 = PSPModule(128)
        self.spp_2 = PSPModule(128)
        self.spp_3 = PSPModule(128)
        self.featureMatch = FeatureMatch()
        self.gap_x = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.gap_fm = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.gap_1_7 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.gap_2_7 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.gap_3_7 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.pre_1 = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU()
        )
        self.linear_a = nn.Linear(1024, 2)

        self.pre_2 = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU()
        )
        self.linear_b = nn.Linear(1024, 2)

        self.pre_3 = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU()
        )
        self.linear_c = nn.Linear(1024, 2)




    def forward(self,x_all,x_crop,t):

        x = self.model(x_all)
        fm,_,_ = self.featureMatch(x_crop, t)   # 考虑对fm进行损失计算
        x = self.gap_x(x)
        fm = self.gap_fm(fm)
        #x = torch.cat([x,fm],1)
        x=x*fm
        # 外围预测
        x_1 = self.spp_1(x)
        x_1 = self.gap_1_7(x_1)
        x_1 = x_1.view(x_1.size(0),-1)
        x_1 = self.pre_1(x_1)
        pre_a = self.linear_a(x_1)

        #中间预测
        x_2 = self.spp_2(x)
        x_2 = self.gap_2_7(x_2)
        x_2 = x_2.view(x_2.size(0), -1)
        x_2 = self.pre_2(x_2)
        pre_b = self.linear_b(x_2)

        #内围预测
        x_3 = self.spp_3(x)
        x_3 = self.gap_3_7(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = self.pre_3(x_3)
        pre_c = self.linear_c(x_3)
        return pre_a,pre_b,pre_c






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
