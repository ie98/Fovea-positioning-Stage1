from torch import nn
from torchvision.models import resnet18
from src.network.SPP.sppmodule import PSPModule
from src.network.v1.fm1.FeatureMatch_1_1 import FeatureMatch


class mainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.resnet18.children())[:6])

        # self.bn_fm = nn.BatchNorm2d(128)
        # self.relu_fm = nn.ReLU(inplace=True)
        self.spp = PSPModule(128)
        self.featureMatch = FeatureMatch()
        self.gap_x = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.gap_fm = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.gap_7 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.pre = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )




    def forward(self,x_all,x_crop,t):
        x = self.model(x_all)
        fm,_,_ = self.featureMatch(x_crop, t)   # 考虑对fm进行损失计算
        # fm = self.bn_fm(fm)
        # fm = self.relu_fm(fm)
        x = self.gap_x(x)
        fm = self.gap_fm(fm)
        # x = torch.cat([x,fm],1)
        x=x*fm
        x = self.spp(x)
        x = self.gap_7(x)
        x = x.view(x.size(0),-1)
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
