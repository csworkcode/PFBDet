"PConv"


class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(dim, ouc, k=1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class PSA_Channel(nn.Module):
    def __init__(self, c1) -> None:
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1)
        self.cv2 = nn.Conv2d(c1, 1, 1)
        self.cv3 = nn.Conv2d(c_, c1, 1)
        self.reshape1 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.reshape2 = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.layernorm = nn.LayerNorm([c1, 1, 1])

    def forward(self, x):  # shape(batch, channel, height, width)
        x1 = self.reshape1(self.cv1(x))  # shape(batch, channel/2, height*width)
        x2 = self.softmax(self.reshape2(self.cv2(x)))  # shape(batch, height*width)
        y = torch.matmul(x1, x2.unsqueeze(-1)).unsqueeze(-1)  # 高维度下的矩阵乘法（最后两个维度相乘）
        return self.sigmoid(self.layernorm(self.cv3(y))) * x


class PSA_Spatial(nn.Module):
    def __init__(self, c1) -> None:
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1)
        self.reshape1 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.globalPooling = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # shape(batch, channel, height, width)
        x1 = self.reshape1(self.cv1(x))  # shape(batch, channel/2, height*width)
        x2 = self.softmax(self.globalPooling(self.cv2(x)).squeeze(-1))  # shape(batch, channel/2, 1)
        y = torch.bmm(x2.permute(0, 2, 1), x1)  # shape(batch, 1, height*width)
        return self.sigmoid(y.view(x.shape[0], 1, x.shape[2], x.shape[3])) * x


class PSA(nn.Module):
    def __init__(self, in_channel, parallel=True) -> None:
        super().__init__()
        self.parallel = parallel
        self.channel = PSA_Channel(in_channel)
        self.spatial = PSA_Spatial(in_channel)

    def forward(self, x):
        if (self.parallel):
            return self.channel(x) + self.spatial(x)
        return self.spatial(self.channel(x))


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        # self.gsb = GSBottleneck(c_, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))