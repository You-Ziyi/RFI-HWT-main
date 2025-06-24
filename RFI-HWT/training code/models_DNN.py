import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class RowAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(RowAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))  # 对每行进行平均池化
        self.max_pool = nn.AdaptiveMaxPool2d((1, None))  # 对每行进行最大池化

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, input_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        # 初始卷积层和ReLU激活函数
        layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))

        # 中间层
        for i in range(depth - 2):
            if i == 11:
                layers.append(
                    nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                              bias=False, dilation=2))
            else:
                layers.append(
                    nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                              bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))

            # 在第5和第11层后添加通道注意力模块
            if i == 4 or i == 10:
                # layers.append(ChannelAttention(n_channels))
                layers.append(RowAttention(n_channels))

            # # 在第6和第12层后添加像素注意力模块
            # if i == 5 or i == 11:
            #     layers.append(PixelAttention(n_channels))

        # 最终卷积层
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=input_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))

        # 组合所有层
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('init weights finished')

