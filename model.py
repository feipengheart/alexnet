#coding=utf-8
import torch
from torch import nn
torch.manual_seed(4)

#构建alexnet网络结构
class Alexnet(nn.Module):
    def __init__(self,nc=10):
        super(Alexnet,self).__init__()
        self.feature=nn.Sequential(
            nn.Conv2d(3,48,11,4,2,bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(3,2),
            nn.Conv2d(48,128,5,1,2,bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(3,2),
            nn.Conv2d(128, 192, 3, 1,1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(192, 192, 3, 1,1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(192, 128, 3, 1,1,bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )
        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4608,2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048,nc)
        )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x=self.feature(x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x

if __name__ == '__main__':
    image=torch.ones((2,3,224,224))
    net=Alexnet()
    print(net(image))
