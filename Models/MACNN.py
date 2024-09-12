from CODE.Utils.package import *


class Classifier_MACNN(nn.Module):
    def __init__(self, input_shape, num_classes, *wargs, **kwargs):
        super().__init__()

        # Define your network layers here
        self.stack1 = self._make_stack(input_shape[1], 64, 2)
        self.pool1 = nn.MaxPool1d(3, 2, padding=1)
        self.stack2 = self._make_stack(192, 128, 2)
        self.pool2 = nn.MaxPool1d(3, 2, padding=1)
        self.stack3 = self._make_stack(384, 256, 2)
        # Add more stacks and pooling layers as required

        self.fc = nn.Linear(
            768, num_classes
        )  # Adjust the input features according to your network

    def forward(self, x):
        x = self.stack1(x)
        x = self.pool1(x)
        x = self.stack2(x)
        x = self.pool2(x)
        x = self.stack3(x)
        # Add more stacks and pooling layers as required

        x = torch.mean(x, 2)  # Global Average Pooling
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

    def _make_stack(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(MACNNBlock(in_channels, out_channels))
            in_channels = out_channels * 3  # 更新 in_channels 为下一层
        return nn.Sequential(*layers)


class MACNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduce=16):
        super(MACNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=6,
            padding="same",
        )
        self.conv3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=12,
            padding="same",
        )
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.attention_fc1 = nn.Linear(out_channels * 3, int(out_channels * 3 / reduce))
        self.attention_fc2 = nn.Linear(int(out_channels * 3 / reduce), out_channels * 3)

    def forward(self, x):
        cov1 = self.conv1(x)
        cov2 = self.conv2(x)
        cov3 = self.conv3(x)

        x = torch.cat([cov1, cov2, cov3], 1)
        x = self.bn(x)
        x = F.relu(x)

        # 注意力机制
        y = torch.mean(x, 2)
        y = F.relu(self.attention_fc1(y))
        y = torch.sigmoid(self.attention_fc2(y))
        y = y.view(
            y.shape[0], y.shape[1], -1
        )  # reshape to [batch_size, out_channels * 3, 1]

        return x * y


# Other block classes (e.g., MCNNBlock, SACNNBlock) should be defined similarly

# Usage
# net = Net(classes_num=10)  # Specify the number of classes
