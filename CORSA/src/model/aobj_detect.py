import torch
import torch.nn.functional as F


# 定义上采样层，邻近插值
class UpsampleLayer(torch.nn.Module):
    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


# 定义卷积层
class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.sub_module(x)


# 定义残差结构
class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),  # 1x1卷积
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1),  # 3x3卷积    #两次卷积完成之后大小不变，  残差后需要还原通道
        )

    def forward(self, x):
        return x + self.sub_module(x)


# 定义下采样层
class DownsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


# 定义卷积块
class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)


# 定义主网络
class Detect_Net(torch.nn.Module):

    def __init__(self):
        super(Detect_Net, self).__init__()

        self.trunk_52 = torch.nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownsamplingLayer(32, 64),
            ResidualLayer(64),
            DownsamplingLayer(64, 128),
            ResidualLayer(128),
            ResidualLayer(128),
            DownsamplingLayer(128, 256),
            ConvolutionalLayer(256,512,3,2,3),
            #DownsamplingLayer(256, 512),
            ResidualLayer(512),  # 8层残差
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),


        #10 最好
        )

        # 做外接口
        self.trunk_26 = torch.nn.Sequential(
            DownsamplingLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),


        #10 最好
        )

        self.trunk_13 = torch.nn.Sequential(
            #DownsamplingLayer(1024, 1024),
            DownsamplingLayer(1024, 2048),
            ResidualLayer(2048),
            ResidualLayer(2048),
            ResidualLayer(2048),
            ResidualLayer(2048),
            ResidualLayer(2048),

        #5最好
        )

        self.convset_13 = torch.nn.Sequential(
            ConvolutionalSet(2048, 1024)
        )

        self.detetion_13 = torch.nn.Sequential(
            ConvolutionalLayer(1024, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 24, 1, 1, 0)  # 24代表3组（3个类别+1个置信度+4个坐标）
        )

        self.up_26 = torch.nn.Sequential(
            ConvolutionalLayer(1024, 1024, 1, 1, 0),  # 上采样得到的图缩小为原来的一半
            UpsampleLayer()
        )

        self.convset_26 = torch.nn.Sequential(
            ConvolutionalSet(2048, 1024)
        )

        self.detetion_26 = torch.nn.Sequential(
            ConvolutionalLayer(1024, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 24, 1, 1, 0)
        )

        self.up_52 = torch.nn.Sequential(
            ConvolutionalLayer(1024, 512, 1, 1, 0),
            UpsampleLayer()
        )

        self.convset_52 = torch.nn.Sequential(
            ConvolutionalSet(1024, 512)
        )

        self.detetion_52 = torch.nn.Sequential(
            ConvolutionalLayer(512, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 24, 1, 1, 0)
        )
        #self.conv1x1 = torch.nn.Sequential(ConvolutionalLayer(1024, 2048, 1, 1, 0))

    def forward(self, x):
        h_52 = self.trunk_52(x)  # 下采样输出

        h_26 = self.trunk_26(h_52)  # 下采样输出
        h_13 = self.trunk_13(h_26)  # 下采样输出
        convset_out_13 = self.convset_13(h_13)

        detetion_out_13 = self.detetion_13(convset_out_13)  # 在13的特征图输出后一部分用来上采样，另一部分用来做侦测
        up_out_26 = self.up_26(convset_out_13)  # 上采样

        route_out_26 = torch.cat((up_out_26, h_26), dim=1)  # concatenate，上采样得到的与下采样得到的在1轴上拼接（只拼接数据，不拼接批次）

        convset_out_26 = self.convset_26(route_out_26)
        detetion_out_26 = self.detetion_26(convset_out_26)
        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)  # 拼接
        #route_out_52 = self.conv1x1(route_out_52)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_out_52 = self.detetion_52(convset_out_52)
        return h_13,route_out_26,route_out_52  # 如果考虑侦测更小或更大的目标可以考虑增加层数


# 测试网络
if __name__ == '__main__':
    net = MainNet()
    x = torch.randn([2, 3, 416, 416], dtype=torch.float32)
    # 测试网络
    y_13, y_26, y_52 = net(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)
    print(y_13.permute([0, 2, 3, 1]).shape)
    print(y_13.view(-1, 13, 13, 3, 8).shape)  # 之所以要进行通道转换是因为标签的shape是NHWC（其中C由两部分组成），变动时可以在网络或者标签中更改

