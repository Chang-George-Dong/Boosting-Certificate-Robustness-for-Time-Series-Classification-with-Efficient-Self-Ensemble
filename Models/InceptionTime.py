from CODE.Utils.package import *


class InceptionModule(nn.Module):
    r"""
    Description: This module represents the Inception block, a key component of the Inception architecture.

    Parameters:
        in_channels: Number of input channels.
        nb_filters: Number of filters for the convolutional layers.
        kernel_size: Size of the kernel for the convolutional layers.
        bottleneck_size (default=32): Size of the bottleneck layer.
        use_bottleneck (default=True): Whether to use the bottleneck layer.
    """

    def __init__(
        self,
        in_channels,
        nb_filters,
        kernel_size,
        bottleneck_size=32,
        use_bottleneck=True,
    ):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = use_bottleneck

        if self.use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_size, kernel_size=1, bias=False
            )
            conv_input_channels = bottleneck_size
            conv_mp_size = bottleneck_size * 4
        else:
            conv_input_channels = in_channels
            conv_mp_size = in_channels

        kernel_size_s = [kernel_size // (2**i) for i in range(3)]
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConstantPad1d(padding=(k // 2 - (k + 1) % 2, k // 2), value=0),
                    nn.Conv1d(
                        conv_input_channels, nb_filters, kernel_size=k, bias=False
                    ),
                )
                for k in kernel_size_s
            ]
        )
        self.conv_mp = nn.Conv1d(conv_mp_size, nb_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        max_pool_1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=(3 - 1) // 2)

        if self.use_bottleneck and x.size(1) > 1:
            x = self.bottleneck(x) if hasattr(self, "bottleneck") else x

        conv_list = [conv(x) for conv in self.convs]

        conv_mp_out = self.conv_mp(max_pool_1)
        conv_list.append(conv_mp_out)

        x = torch.cat(conv_list, dim=1)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Classifier_INCEPTION(nn.Module):
    r"""
    Description: This is the main classifier that uses multiple InceptionModule blocks.

    Parameters:
        input_shape: Shape of the input data.
        nb_classes (default=2): Number of output classes.
        nb_filters (default=32): Number of filters for the convolutional layers.
        use_residual (default=True): Whether to use residual connections.
        use_bottleneck (default=True): Whether to use the bottleneck layer in the Inception blocks.
        depth (default=6): Depth of the network, i.e., number of InceptionModule blocks.
        kernel_size (default=41): Size of the kernel for the convolutional layers in the InceptionModule blocks.
        Usage:

        Importing the necessary components:
            from classifier import InceptionModule, InceptionClassifier

        Creating an Inception block:
            `inception_block = InceptionModule(in_channels=1, nb_filters=32, kernel_size=41)`

        Creating the Inception classifier:
            `classifier = InceptionClassifier(input_shape=(None, 1, 128), nb_classes=10)`

        Forward pass:
            output = classifier(input_tensor)

        Notes:
            Ensure that the Package module is accessible or available in the directory structure, as it's imported at the beginning of classifier.py.
            The softmax activation function is applied at the end of the InceptionClassifier forward pass, so the output will be in the range [0, 1] and sum up to 1 across classes.

    """

    def __init__(
        self,
        # nb_classes=2,
        # input_shape=(None, 1, 128),
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        depth=6,
        kernel_size=41,
        **kwargs
    ):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.use_residual = use_residual
        self.depth = depth
        self.nb_filters = nb_filters

        # Define shortcut layers
        self.shortcut_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.nb_filters * 4 if i else 1,
                    self.nb_filters * 4,
                    kernel_size=1,
                    bias=False,
                )
                for i in range(depth // 3)
            ]
        )
        self.shortcut_bns = nn.ModuleList(
            [nn.BatchNorm1d(self.nb_filters * 4) for _ in range(depth // 3)]
        )
        self.shortcut_relus = nn.ModuleList([nn.ReLU() for _ in range(depth // 3)])

        # Adjust the in_channels according to the input shape
        self.in_channels = self.input_shape[1]
        self.layers = nn.ModuleList()

        for d in range(self.depth):
            self.layers.append(
                InceptionModule(
                    self.in_channels,
                    nb_filters,
                    kernel_size - 1,
                    use_bottleneck=use_bottleneck,
                )
            )
            self.in_channels = nb_filters * 4

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def short_cut_layer(self, x, input_res, shortcut_idx):
        x_ = self.shortcut_convs[shortcut_idx](input_res)
        x_ = self.shortcut_bns[shortcut_idx](x_)
        x_new = x + x_
        x_new = self.shortcut_relus[shortcut_idx](x_new)
        return x_new

    def forward(self, x):
        input_res = x
        shortcut_idx = 0

        for d, layer in enumerate(self.layers):
            x = layer(x)

            if self.use_residual and d % 3 == 2:
                x = self.short_cut_layer(x, input_res, shortcut_idx)
                shortcut_idx += 1
                input_res = x

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
