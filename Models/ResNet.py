from CODE.Utils.package import *


class ResRoad(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResRoad, self).__init__()
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.downsample(x)


class MainRoad(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
    ):
        super(MainRoad, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResNetBlock, self).__init__()
        self.res = ResRoad(in_channels, out_channels, stride)

        self.layers = nn.ModuleList()
        self.layers.append(
            MainRoad(in_channels, out_channels, kernel_size, stride, padding)
        )
        self.layers.append(
            MainRoad(out_channels, out_channels, kernel_size, 1, padding)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.res(x)
        for layer in self.layers:
            x = layer(x)
        x += identity
        x = self.relu(x)
        return x


class ResNetBlock_deep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResNetBlock_deep, self).__init__()
        self.res = ResRoad(in_channels, out_channels)

        self.layers = nn.ModuleList()
        self.layers.append(MainRoad(in_channels, out_channels, 1, 1, 0))
        self.layers.append(
            MainRoad(out_channels, out_channels, kernel_size, stride, padding)
        )
        self.layers.append(MainRoad(out_channels, out_channels, 1, 1, 0))

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.res(x)
        for layer in self.layers:
            x = layer(x)
        x += identity
        x = self.relu(x)
        return x


class Classifier_ResNet(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        channels_list,
        kernel_size_list,
        stride_list,
        deep=False,
        **kwargs
    ):
        print(kwargs)
        super().__init__()
        input_channels = input_shape[1]

        self.start = MainRoad(
            input_channels,
            channels_list[0],
            kernel_size=kernel_size_list[0],
            stride=stride_list[0],
            padding=(kernel_size_list[0] - 1) // 2,
        )

        self.resblock = ResNetBlock_deep if deep else ResNetBlock
        self.layers = nn.ModuleList()

        for idx in range(len(channels_list) - 1):
            in_channels = channels_list[idx]
            out_channels = channels_list[idx + 1]
            self.layers.append(
                self.resblock(
                    in_channels,
                    out_channels,
                    kernel_size_list[idx],
                    stride=stride_list[idx + 1],
                    padding=(kernel_size_list[idx] - 1) // 2,
                )
            )
            in_channels = out_channels  # 更新输入通道数

        # 全局平均池化和全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels_list[-1], num_classes)

    def forward(self, x):
        x = self.start(x)

        for layer in self.layers:
            x = layer(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


class Classifier_ResNet18(Classifier_ResNet):
    def __init__(self, input_shape, num_classes, **kwargs):
        # 经典的ResNet18在每个块中通道数翻倍
        channels_list = [64, 64, 128, 128, 256, 256, 512, 512]
        kernel_size_list = [7, *[3] * 7]  # 根据需要调整核大小
        stride_list = [2, 1, 1, 2, 1, 2, 1, 2, 1]

        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            channels_list=channels_list,
            kernel_size_list=kernel_size_list,
            stride_list=stride_list,  # 可能需要调整
            deep=False,  # ResNet18不使用bottleneck结构
            **kwargs
        )
