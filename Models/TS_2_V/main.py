from CODE.Utils.package import *


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv2 = SamePadConv(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


class TSEncoder(nn.Module):
    def __init__(
        self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode="binomial"
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    @staticmethod
    def generate_continuous_mask(B, T, n=5, l=0.1):
        res = torch.full((B, T), True, dtype=torch.bool)
        if isinstance(n, float):
            n = int(n * T)
        n = max(min(n, T // 2), 1)

        if isinstance(l, float):
            l = int(l * T)
        l = max(l, 1)

        for i in range(B):
            for _ in range(n):
                t = np.random.randint(T - l + 1)
                res[i, t : t + l] = False
        return res

    @staticmethod
    def generate_binomial_mask(B, T, p=0.5):
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        # [16, 1, 1460]
        x[~nan_mask] = 0
        # nn.Linear 层期望其输入的最后一个维度是 input_dims
        x = self.input_fc(x)  # B x T x Ch
        # 根据注释 # B x T x Ch，看起来 self.input_fc(x) 期望的输入形状应该是 [batch_size, timeseries_length, channels]，这里是 [16, 1460, 1]。然而，这与 nn.Linear 的标准输入形状不符。nn.Linear 通常期望的输入形状是 [batch_size, *, features]，其中 features 需要与 nn.Linear 的 input_dims 匹配。
        # David 2024年1月25日： 其实我是错的，它这部分代码的一开始期望的输入形状就是 [batch_size, timeseries_length, channels]，这里的 input_dims 是 channels。

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"

        if mask == "binomial":
            mask = self.generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = self.generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x


class TS2Vec(nn.Module):
    """The TS2Vec model"""

    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
    ):
        """Initialize a TS2Vec model.

        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        """

        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self._net = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
        )
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0

    @staticmethod
    def torch_pad_nan(arr, left=0, right=0, dim=0):
        if left > 0:
            padshape = list(arr.shape)
            padshape[dim] = left
            arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
        if right > 0:
            padshape = list(arr.shape)
            padshape[dim] = right
            arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
        return arr

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x, mask)
        if encoding_window == "full_series":
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == "multiscale":
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p,
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(
        self,
        data,
        mask=None,
        encoding_window=None,
        causal=False,
        sliding_length=None,
        sliding_padding=0,
        batch_size=None,
    ):
        """Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        """
        assert self.net is not None, "please train or load a net first"
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        # David: I think this is not necessary
        # Check if data is a numpy array and convert to tensor if necessary
        # if isinstance(data, np.ndarray):
        #     data = torch.from_numpy(data).to(torch.float)
        # elif not isinstance(data, torch.Tensor):
        #     raise TypeError("Data must be a np.ndarray or a torch.Tensor")

        # Create DataLoader from TensorDataset
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size)

        # dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        # loader = DataLoader(dataset, batch_size=batch_size)

        # 攻击需要梯度
        # with torch.no_grad():
        output = []
        for batch in loader:
            x = batch[0]
            if sliding_length is not None:
                reprs = []
                if n_samples < batch_size:
                    calc_buffer = []
                    calc_buffer_l = 0
                for i in range(0, ts_l, sliding_length):
                    l = i - sliding_padding
                    r = i + sliding_length + (sliding_padding if not causal else 0)
                    x_sliding = self.torch_pad_nan(
                        x[:, max(l, 0) : min(r, ts_l)],
                        left=-l if l < 0 else 0,
                        right=r - ts_l if r > ts_l else 0,
                        dim=1,
                    )
                    if n_samples < batch_size:
                        if calc_buffer_l + n_samples > batch_size:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(
                                    sliding_padding,
                                    sliding_padding + sliding_length,
                                ),
                                encoding_window=encoding_window,
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                        calc_buffer.append(x_sliding)
                        calc_buffer_l += n_samples
                    else:
                        out = self._eval_with_pooling(
                            x_sliding,
                            mask,
                            slicing=slice(
                                sliding_padding, sliding_padding + sliding_length
                            ),
                            encoding_window=encoding_window,
                        )
                        reprs.append(out)

                if n_samples < batch_size:
                    if calc_buffer_l > 0:
                        out = self._eval_with_pooling(
                            torch.cat(calc_buffer, dim=0),
                            mask,
                            slicing=slice(
                                sliding_padding, sliding_padding + sliding_length
                            ),
                            encoding_window=encoding_window,
                        )
                        reprs += torch.split(out, n_samples)
                        calc_buffer = []
                        calc_buffer_l = 0

                out = torch.cat(reprs, dim=1)
                if encoding_window == "full_series":
                    out = F.max_pool1d(
                        out.transpose(1, 2).contiguous(),
                        kernel_size=out.size(1),
                    ).squeeze(1)
            else:
                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                if encoding_window == "full_series":
                    out = out.squeeze(1)

            output.append(out)

        output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output


class Classifier_TS2V(nn.Module):
    def __init__(self, input_shape, num_classes, TS2Vec_P=dict(), **kwargs):
        super().__init__()
        self.input_dims = input_shape[1]
        for k, v in kwargs.items():
            setattr(self, k, v)

        # This output_dims is the output_dims of TS2Vec,
        # not the output_dims of MLP.
        self.output_dims = kwargs.get("output_dims", 320)

        self.__init_ts2vec__(TS2Vec_P)
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dims, 320),
            nn.ReLU(),
            nn.BatchNorm1d(320),
        )
        # 定义最后的线性层
        self.linear = nn.Linear(320, num_classes)
        self.to(self.device)

    def __init_ts2vec__(self, TS2Vec_P) -> None:
        self.ts2vec_block = TS2Vec(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            **TS2Vec_P,
        )

        default_path = os.path.join(
            os.path.dirname(__file__), "train_state", "UCR", self.dataset
        )
        path = TS2Vec_P.get("state_dict_path", default_path)
        state_dict = torch.load(path, map_location=self.device)
        self.ts2vec_block.net.load_state_dict(state_dict)
        self.ts2vec_block.to(self.device)
        self.ts2vec_block.net.to(self.device)

        if self.state.lower() == "train":
            for param in self.ts2vec_block._net.parameters():
                param.requires_grad = False
        elif self.state.lower() == "attack":
            for param in self.ts2vec_block._net.parameters():
                param.requires_grad = True
        self.ts2vec_block.net.eval()

    def encode_layer(self, x):
        x = self.ts2vec_block.encode(x, encoding_window="full_series").to(self.device)
        return x

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.encode_layer(x)
        x = self.mlp(x)
        logits = self.linear(x)
        # probabilities = F.softmax(logits, dim=1)
        return logits
