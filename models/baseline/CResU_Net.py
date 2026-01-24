import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_size=3, padding=1, kernels_per_layer=2):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class Residual1(nn.Module):
    def __init__(self, in_channel, out_channel,mid_channel=None,strides=1):
        super(Residual1, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.out_channel =out_channel
        self.strides=strides
        self.branch = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3,  padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.final=nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        out=self.branch(x)
        out= out+x
        out= self.final(out)
        return out


class Residual4(nn.Module):
    def __init__(self, in_channel, out_channel,mid_channel=None,strides=1):
        super(Residual4, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.out_channel =out_channel
        self.strides=strides
        self.branch = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=1,  padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.final = nn.Sequential(
            nn.Conv2d(in_channel + out_channel, out_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out=self.branch(x)
        out = torch.cat([out, x], dim=1)
        out= self.final(out)
        return out


class Residual2(nn.Module):
    def __init__(self, in_channel, out_channel,mid_channel=None,strides=1):
        super(Residual2, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.out_channel =out_channel
        self.strides=strides
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
        )
        self.final=nn.Sequential(
            nn.Conv2d(mid_channel+out_channel, out_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        out = torch.cat([branch1, branch2], dim=1)
        out= self.final(out)
        return out


class Residual3(nn.Module):
    def __init__(self, in_channel, out_channel,mid_channel=None,strides=1):
        super(Residual3, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.out_channel =out_channel
        self.strides=strides
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, bias=False,padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, bias=False,padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, bias=False,dilation=2,padding=2),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
        )
        self.final=nn.Sequential(
            nn.Conv2d(mid_channel+out_channel, out_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        out = torch.cat([branch1, branch2], dim=1)
        out= self.final(out)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=None):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(in_channels, out_channels,kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False,kernel_size=None):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DepthwiseSeparableConv(in_channels, out_channels,kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DepthwiseSeparableConv(in_channels, out_channels,kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class CRUNet(nn.Module):
    def __init__(self, selected_dim, in_channels, out_channels, device):
        super(CRUNet, self).__init__()

        self.n_channels = in_channels
        self.out_channels = out_channels
        self.selected_dim = selected_dim

        self.inc = DepthwiseSeparableConv(in_channels=in_channels, output_channels=64)
        self.res1= nn.Sequential(
            Residual1(in_channel=64, out_channel=64,mid_channel=32),
            DepthwiseSeparableConv(in_channels=64, output_channels=64)
        )

        self.ca1 = CoordAtt(inp=64, oup=64)

        self.down1 = Down(in_channels=64,out_channels=128,kernel_size=3)
        self.ca2 = CoordAtt(inp=128, oup=128)
        self.res2 = nn.Sequential(
            Residual2(in_channel=128, out_channel=128, mid_channel=64),
            DepthwiseSeparableConv(in_channels=128, output_channels=128)
        )
        self.ca3 = CoordAtt(inp=256, oup=256)
        self.down2 = Down(in_channels=128, out_channels=256,kernel_size=3)
        self.res3 = nn.Sequential(
            Residual3(in_channel=256, out_channel=256, mid_channel=128),
            DepthwiseSeparableConv(in_channels=256, output_channels=256)
        )
        self.ca4 = CoordAtt(inp=512, oup=512)
        self.down3 = Down(in_channels=256, out_channels=512,kernel_size=3)

        self.res4 = nn.Sequential(
            Residual4(in_channel=512, out_channel=512, mid_channel=256),
            DepthwiseSeparableConv(in_channels=512, output_channels=512,kernel_size=3)
        )
        self.ca5 = CoordAtt(inp=1024, oup=1024)
        self.down4 = Down(in_channels=512, out_channels=1024,kernel_size=3)
        self.up1 = Up(in_channels=1024, out_channels=512,kernel_size=3)
        self.res5 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=1024, output_channels=512,kernel_size=3),
            Residual4(in_channel=512, out_channel=512, mid_channel=256)
        )
        self.up2 = Up(in_channels=512, out_channels=256,kernel_size=3)
        self.res6 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=512, output_channels=256),
            Residual3(in_channel=256, out_channel=256, mid_channel=128)
        )
        self.up3 = Up(in_channels=256, out_channels=128,kernel_size=3)
        self.res7 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=256, output_channels=128),
            Residual2(in_channel=128, out_channel=128, mid_channel=64)
        )
        self.up4 = Up(in_channels=128, out_channels=64,kernel_size=3)
        self.res8 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=128, output_channels=64),
            Residual1(in_channel=64, out_channel=64, mid_channel=32)
        )
        self.out = DepthwiseSeparableConv(in_channels=64, output_channels=out_channels)
        self.is_trainable = True

    def forward(self, x, **kwargs):

        x = x[:, :, self.selected_dim]

        x1 = self.inc(x)
        x2 = self.res1(x1)
        ca1 = self.ca1(x2)
        x3 = self.down1(x2)
        x4 = self.res2(x3)
        ca2 = self.ca2(x4)
        x5 = self.down2(x4)
        x6 = self.res3(x5)
        ca3 = self.ca3(x6)
        x7 = self.down3(x6)
        x8 = self.res4(x7)
        ca4 = self.ca4(x8)
        x9 = self.down4(x8)
        ca5 = self.ca5(x9)
        x10 = self.up1(ca5, ca4)
        x11 = self.res5(x10)
        x12 = self.up2(x11, ca3)
        x13 = self.res6(x12)
        x14 = self.up3(x13, ca2)
        x15 = self.res7(x14)
        x16 = self.up4(x15, ca1)
        x17 = self.res8(x16)
        out = self.out(x17)

        output = out.unsqueeze(2)

        return output

