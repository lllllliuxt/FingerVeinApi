import torch.nn as nn
import torch.nn.functional as F
import torch
# dropout比例
prob = 0.5
# 分类数
num_classes = 600

# 搭建网络框架
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # in：60*175 out: 55*170
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.LocalResponseNorm(5)
        )
        # in：55*170 out: 50*165
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
        )
        # in：50*165 out: 45*160
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
        )
        # in：45*160 out: 40*155
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
        )
        # in：40*155 out: 35*150
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
        )
        # in 35*150=5250 out:150
        self.F1 = nn.Sequential(
            nn.Linear(in_features=336000, out_features=150),
            nn.Dropout(p=prob)
        )
        # 全北数据集分为600类
        self.F2 = nn.Linear(in_features=150, out_features=num_classes)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        x = torch.flatten(x, 1)
        # print(np.shape(x))
        x = self.F1(x)
        output = self.F2(x)

        return output 