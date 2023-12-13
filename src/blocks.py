import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# -----------------------------backbone----------------------------- #
def down_sampling(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)


def up_sampling(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super(ConvBNReLU, self).__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RSU7(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(RSU7, self).__init__()

        self.in_out = ConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = ConvBNReLU(in_ch=out_ch, out_ch=mid_ch)
        self.mid_mid = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch)
        self.mid2_mid = ConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch)
        self.mid_out = ConvBNReLU(in_ch=mid_ch, out_ch=out_ch)
        self.mid2_out = ConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)
        self.mid_mid_dilation = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res2 = x  # [1, 16, 144, 144]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res3 = x  # [1, 16, 72, 72]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res4 = x  # [1, 16, 36, 36]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res5 = x  # [1, 16, 18, 18]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res6 = x  # [1, 16, 9, 9]

        x = self.mid_mid_dilation(x)  # [1, 16, 9, 9]

        x = self.mid2_mid(torch.cat((x, x_res6), dim=1))
        x = up_sampling(x, x_res5)  # [1, 16, 18, 18]

        x = self.mid2_mid(torch.cat((x, x_res5), dim=1))
        x = up_sampling(x, x_res4)  # [1, 16, 36, 36]

        x = self.mid2_mid(torch.cat((x, x_res4), dim=1))
        x = up_sampling(x, x_res3)  # [1, 16, 72, 72]

        x = self.mid2_mid(torch.cat((x, x_res3), dim=1))
        x = up_sampling(x, x_res2)  # [1, 16, 144, 144]

        x = self.mid2_mid(torch.cat((x, x_res2), dim=1))
        x = up_sampling(x, x_res1)  # [1, 16, 288, 288]

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


class RSU6(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(RSU6, self).__init__()

        self.in_out = ConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = ConvBNReLU(in_ch=out_ch, out_ch=mid_ch)
        self.mid_mid = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch)
        self.mid2_mid = ConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch)
        self.mid_out = ConvBNReLU(in_ch=mid_ch, out_ch=out_ch)
        self.mid2_out = ConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)
        self.mid_mid_dilation = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res2 = x  # [1, 16, 144, 144]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res3 = x  # [1, 16, 72, 72]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res4 = x  # [1, 16, 36, 36]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res5 = x  # [1, 16, 18, 18]

        x = self.mid_mid_dilation(x)  # [1, 16, 18, 18]

        x = self.mid2_mid(torch.cat((x, x_res5), dim=1))
        x = up_sampling(x, x_res4)  # [1, 16, 36, 36]

        x = self.mid2_mid(torch.cat((x, x_res4), dim=1))
        x = up_sampling(x, x_res3)  # [1, 16, 72, 72]

        x = self.mid2_mid(torch.cat((x, x_res3), dim=1))
        x = up_sampling(x, x_res2)  # [1, 16, 144, 144]

        x = self.mid2_mid(torch.cat((x, x_res2), dim=1))
        x = up_sampling(x, x_res1)  # [1, 16, 288, 288]

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


class RSU5(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(RSU5, self).__init__()

        self.in_out = ConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = ConvBNReLU(in_ch=out_ch, out_ch=mid_ch)
        self.mid_mid = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch)
        self.mid2_mid = ConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch)
        self.mid_out = ConvBNReLU(in_ch=mid_ch, out_ch=out_ch)
        self.mid2_out = ConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)
        self.mid_mid_dilation = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res2 = x  # [1, 16, 144, 144]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res3 = x  # [1, 16, 72, 72]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res4 = x  # [1, 16, 36, 36]

        x = self.mid_mid_dilation(x)  # [1, 16, 36, 36]

        x = self.mid2_mid(torch.cat((x, x_res4), dim=1))
        x = up_sampling(x, x_res3)  # [1, 16, 72, 72]

        x = self.mid2_mid(torch.cat((x, x_res3), dim=1))
        x = up_sampling(x, x_res2)  # [1, 16, 144, 144]

        x = self.mid2_mid(torch.cat((x, x_res2), dim=1))
        x = up_sampling(x, x_res1)  # [1, 16, 288, 288]

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


class RSU4(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(RSU4, self).__init__()

        self.in_out = ConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = ConvBNReLU(in_ch=out_ch, out_ch=mid_ch)
        self.mid_mid = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch)
        self.mid2_mid = ConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch)
        self.mid_out = ConvBNReLU(in_ch=mid_ch, out_ch=out_ch)
        self.mid2_out = ConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)
        self.mid_mid_dilation = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res2 = x  # [1, 16, 144, 144]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res3 = x  # [1, 16, 72, 72]

        x = self.mid_mid_dilation(x)  # [1, 16, 36, 36]

        x = self.mid2_mid(torch.cat((x, x_res3), dim=1))
        x = up_sampling(x, x_res2)  # [1, 16, 144, 144]

        x = self.mid2_mid(torch.cat((x, x_res2), dim=1))
        x = up_sampling(x, x_res1)  # [1, 16, 288, 288]

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(RSU4F, self).__init__()

        self.in_out = ConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = ConvBNReLU(in_ch=out_ch, out_ch=mid_ch)

        self.mid_mid_dilation2 = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)
        self.mid_mid_dilation4 = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=4)
        self.mid_mid_dilation8 = ConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=8)

        self.mid2_mid_dilation4 = ConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch, dilation=4)
        self.mid2_mid_dilation2 = ConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch, dilation=2)
        self.mid2_out = ConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = self.mid_mid_dilation2(x)
        x_res2 = x  # [1, 16, 288, 288]

        x = self.mid_mid_dilation4(x)
        x_res3 = x  # [1, 16, 288, 288]

        x = self.mid_mid_dilation8(x)

        x = self.mid2_mid_dilation4(torch.cat((x, x_res3), dim=1))

        x = self.mid2_mid_dilation2(torch.cat((x, x_res2), dim=1))

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


# -----------------------------improvement 1----------------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.se = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_result = self.max_pool(x)
        avg_result = self.avg_pool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        return self.sigmoid(max_out + avg_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_result, _ = torch.max(input=x, dim=1, keepdim=True)
        avg_result = torch.mean(input=x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        return self.sigmoid(self.conv(result))


class CBAMBlock(nn.Module):
    def __init__(self, channel: int):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channel=channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x) * x
        return self.spatial_attention(x) * x


# -----------------------------improvement 2----------------------------- #
class DepthWiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DepthWiseConv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                    stride=1, padding=1, groups=in_channels, bias=False)

        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                    stride=1, padding=0, groups=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class DWConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super(DWConvBNReLU, self).__init__()

        self.conv = DepthWiseConv(in_channels=in_ch, out_channels=out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DWRSU7(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(DWRSU7, self).__init__()

        self.in_out = DWConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = DWConvBNReLU(in_ch=out_ch, out_ch=mid_ch)
        self.mid_mid = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch)
        self.mid2_mid = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch)
        self.mid_out = DWConvBNReLU(in_ch=mid_ch, out_ch=out_ch)
        self.mid2_out = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)
        self.mid_mid_dilation = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res2 = x  # [1, 16, 144, 144]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res3 = x  # [1, 16, 72, 72]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res4 = x  # [1, 16, 36, 36]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res5 = x  # [1, 16, 18, 18]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res6 = x  # [1, 16, 9, 9]

        x = self.mid_mid_dilation(x)  # [1, 16, 9, 9]

        x = self.mid2_mid(torch.cat((x, x_res6), dim=1))
        x = up_sampling(x, x_res5)  # [1, 16, 18, 18]

        x = self.mid2_mid(torch.cat((x, x_res5), dim=1))
        x = up_sampling(x, x_res4)  # [1, 16, 36, 36]

        x = self.mid2_mid(torch.cat((x, x_res4), dim=1))
        x = up_sampling(x, x_res3)  # [1, 16, 72, 72]

        x = self.mid2_mid(torch.cat((x, x_res3), dim=1))
        x = up_sampling(x, x_res2)  # [1, 16, 144, 144]

        x = self.mid2_mid(torch.cat((x, x_res2), dim=1))
        x = up_sampling(x, x_res1)  # [1, 16, 288, 288]

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


class DWRSU6(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(DWRSU6, self).__init__()

        self.in_out = DWConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = DWConvBNReLU(in_ch=out_ch, out_ch=mid_ch)
        self.mid_mid = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch)
        self.mid2_mid = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch)
        self.mid_out = DWConvBNReLU(in_ch=mid_ch, out_ch=out_ch)
        self.mid2_out = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)
        self.mid_mid_dilation = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res2 = x  # [1, 16, 144, 144]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res3 = x  # [1, 16, 72, 72]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res4 = x  # [1, 16, 36, 36]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res5 = x  # [1, 16, 18, 18]

        x = self.mid_mid_dilation(x)  # [1, 16, 18, 18]

        x = self.mid2_mid(torch.cat((x, x_res5), dim=1))
        x = up_sampling(x, x_res4)  # [1, 16, 36, 36]

        x = self.mid2_mid(torch.cat((x, x_res4), dim=1))
        x = up_sampling(x, x_res3)  # [1, 16, 72, 72]

        x = self.mid2_mid(torch.cat((x, x_res3), dim=1))
        x = up_sampling(x, x_res2)  # [1, 16, 144, 144]

        x = self.mid2_mid(torch.cat((x, x_res2), dim=1))
        x = up_sampling(x, x_res1)  # [1, 16, 288, 288]

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


class DWRSU5(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(DWRSU5, self).__init__()

        self.in_out = DWConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = DWConvBNReLU(in_ch=out_ch, out_ch=mid_ch)
        self.mid_mid = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch)
        self.mid2_mid = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch)
        self.mid_out = DWConvBNReLU(in_ch=mid_ch, out_ch=out_ch)
        self.mid2_out = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)
        self.mid_mid_dilation = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res2 = x  # [1, 16, 144, 144]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res3 = x  # [1, 16, 72, 72]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res4 = x  # [1, 16, 36, 36]

        x = self.mid_mid_dilation(x)  # [1, 16, 36, 36]

        x = self.mid2_mid(torch.cat((x, x_res4), dim=1))
        x = up_sampling(x, x_res3)  # [1, 16, 72, 72]

        x = self.mid2_mid(torch.cat((x, x_res3), dim=1))
        x = up_sampling(x, x_res2)  # [1, 16, 144, 144]

        x = self.mid2_mid(torch.cat((x, x_res2), dim=1))
        x = up_sampling(x, x_res1)  # [1, 16, 288, 288]

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


class DWRSU4(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(DWRSU4, self).__init__()

        self.in_out = DWConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = DWConvBNReLU(in_ch=out_ch, out_ch=mid_ch)
        self.mid_mid = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch)
        self.mid2_mid = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch)
        self.mid_out = DWConvBNReLU(in_ch=mid_ch, out_ch=out_ch)
        self.mid2_out = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)
        self.mid_mid_dilation = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res2 = x  # [1, 16, 144, 144]

        x = down_sampling(x)
        x = self.mid_mid(x)
        x_res3 = x  # [1, 16, 72, 72]

        x = self.mid_mid_dilation(x)  # [1, 16, 36, 36]

        x = self.mid2_mid(torch.cat((x, x_res3), dim=1))
        x = up_sampling(x, x_res2)  # [1, 16, 144, 144]

        x = self.mid2_mid(torch.cat((x, x_res2), dim=1))
        x = up_sampling(x, x_res1)  # [1, 16, 288, 288]

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


class DWRSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(DWRSU4F, self).__init__()

        self.in_out = DWConvBNReLU(in_ch=in_ch, out_ch=out_ch)
        self.out_mid = DWConvBNReLU(in_ch=out_ch, out_ch=mid_ch)

        self.mid_mid_dilation2 = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=2)
        self.mid_mid_dilation4 = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=4)
        self.mid_mid_dilation8 = DWConvBNReLU(in_ch=mid_ch, out_ch=mid_ch, dilation=8)

        self.mid2_mid_dilation4 = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch, dilation=4)
        self.mid2_mid_dilation2 = ConvBNReLU(in_ch=mid_ch * 2, out_ch=mid_ch, dilation=2)
        self.mid2_out = DWConvBNReLU(in_ch=mid_ch * 2, out_ch=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_out(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = self.out_mid(x)
        x_res1 = x  # [1, 16, 288, 288]

        x = self.mid_mid_dilation2(x)
        x_res2 = x  # [1, 16, 288, 288]

        x = self.mid_mid_dilation4(x)
        x_res3 = x  # [1, 16, 288, 288]

        x = self.mid_mid_dilation8(x)

        x = self.mid2_mid_dilation4(torch.cat((x, x_res3), dim=1))

        x = self.mid2_mid_dilation2(torch.cat((x, x_res2), dim=1))

        x = self.mid2_out(torch.cat((x, x_res1), dim=1))

        return x + x_res0


# -----------------------------U-Net part----------------------------- #
class DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, mid_ch=None):
        super(DoubleConv, self).__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


if __name__ == "__main__":
    model = DoubleConv(3, 64)
    summary(model.to(torch.device('cuda')), (3, 288, 288))
    