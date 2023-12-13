import torch
import torch.nn as nn
# import thop
from torchsummary import summary

from src.blocks import CBAMBlock  # improvement 1 -> attention part
from src.blocks import DWRSU7, DWRSU6, DWRSU5, DWRSU4, DWRSU4F  # improvement 2 -> depth_wise part
from src.blocks import DoubleConv  # UNet part
from src.blocks import up_sampling, down_sampling, RSU7, RSU6, RSU5, RSU4, RSU4F


# -----------------------------U^2Net----------------------------- #
class U2Net(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(U2Net, self).__init__()

        # encoder part
        self.encoder1 = RSU7(in_ch=in_ch, mid_ch=32, out_ch=64)
        self.encoder2 = RSU6(in_ch=64, mid_ch=32, out_ch=128)
        self.encoder3 = RSU5(in_ch=128, mid_ch=64, out_ch=256)
        self.encoder4 = RSU4(in_ch=256, mid_ch=128, out_ch=512)
        self.encoder5 = RSU4F(in_ch=512, mid_ch=256, out_ch=512)
        self.encoder6 = RSU4F(in_ch=512, mid_ch=256, out_ch=512)

        # decoder part
        self.decoder5 = RSU4F(in_ch=1024, mid_ch=256, out_ch=512)
        self.decoder4 = RSU4(in_ch=1024, mid_ch=128, out_ch=256)
        self.decoder3 = RSU5(in_ch=512, mid_ch=64, out_ch=128)
        self.decoder2 = RSU6(in_ch=256, mid_ch=32, out_ch=64)
        self.decoder1 = RSU7(in_ch=128, mid_ch=16, out_ch=64)

        # side part
        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=256, out_channels=out_ch, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)

        # output part
        self.outputs = nn.Conv2d(in_channels=out_ch * 6, out_channels=out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = down_sampling(x)
        x = self.encoder2(x)
        x_res1 = x  # [1, 128, 144, 144]

        x = down_sampling(x)
        x = self.encoder3(x)
        x_res2 = x  # [1, 256, 72, 72]

        x = down_sampling(x)
        x = self.encoder4(x)
        x_res3 = x  # [1, 512, 36, 36]

        x = down_sampling(x)
        x = self.encoder5(x)
        x_res4 = x  # [1, 512, 18, 18]

        x = down_sampling(x)
        x = self.encoder6(x)  # [1, 512, 9, 9]

        x = up_sampling(x, x_res4)  # [1, 512, 18, 18]
        outputs6 = self.side6(x)

        x = self.decoder5(torch.cat((x, x_res4), dim=1))
        outputs5 = self.side5(x)
        x = up_sampling(x, x_res3)  # [1, 512, 36, 36]

        x = self.decoder4(torch.cat((x, x_res3), dim=1))
        outputs4 = self.side4(x)
        x = up_sampling(x, x_res2)  # [1, 256, 72, 72]

        x = self.decoder3(torch.cat((x, x_res2), dim=1))
        outputs3 = self.side3(x)
        x = up_sampling(x, x_res1)  # [1, 128, 144, 144]

        x = self.decoder2(torch.cat((x, x_res1), dim=1))
        outputs2 = self.side2(x)
        x = up_sampling(x, x_res0)  # [1, 64, 288, 288]

        x = self.decoder1(torch.cat((x, x_res0), dim=1))  # [1, 64, 288, 288]
        outputs1 = self.side1(x)

        outputs2 = up_sampling(outputs2, outputs1)
        outputs3 = up_sampling(outputs3, outputs1)
        outputs4 = up_sampling(outputs4, outputs1)
        outputs5 = up_sampling(outputs5, outputs1)
        outputs6 = up_sampling(outputs6, outputs1)

        outputs = self.outputs(torch.cat((outputs1, outputs2, outputs3, outputs4, outputs5, outputs6), dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [outputs, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6]
        else:
            return torch.sigmoid(outputs)


# -----------------------------U^2Net_lite----------------------------- #
class U2Net_lite(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(U2Net_lite, self).__init__()

        # encoder part
        self.encoder1 = RSU7(in_ch=in_ch, mid_ch=16, out_ch=64)
        self.encoder2 = RSU6(in_ch=64, mid_ch=16, out_ch=64)
        self.encoder3 = RSU5(in_ch=64, mid_ch=16, out_ch=64)
        self.encoder4 = RSU4(in_ch=64, mid_ch=16, out_ch=64)
        self.encoder5 = RSU4F(in_ch=64, mid_ch=16, out_ch=64)
        self.encoder6 = RSU4F(in_ch=64, mid_ch=16, out_ch=64)

        # decoder part
        self.decoder5 = RSU4F(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder4 = RSU4(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder3 = RSU5(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder2 = RSU6(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder1 = RSU7(in_ch=128, mid_ch=16, out_ch=64)

        # side part
        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)

        # output part
        self.outputs = nn.Conv2d(in_channels=out_ch * 6, out_channels=out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = down_sampling(x)
        x = self.encoder2(x)
        x_res1 = x  # [1, 64, 144, 144]

        x = down_sampling(x)
        x = self.encoder3(x)
        x_res2 = x  # [1, 64, 72, 72]

        x = down_sampling(x)
        x = self.encoder4(x)
        x_res3 = x  # [1, 64, 36, 36]

        x = down_sampling(x)
        x = self.encoder5(x)
        x_res4 = x  # [1, 64, 18, 18]

        x = down_sampling(x)
        x = self.encoder6(x)  # [1, 64, 9, 9]

        x = up_sampling(x, x_res4)  # [1, 64, 18, 18]
        outputs6 = self.side6(x)

        x = self.decoder5(torch.cat((x, x_res4), dim=1))
        outputs5 = self.side5(x)
        x = up_sampling(x, x_res3)  # [1, 64, 36, 36]

        x = self.decoder4(torch.cat((x, x_res3), dim=1))
        outputs4 = self.side4(x)
        x = up_sampling(x, x_res2)  # [1, 64, 72, 72]

        x = self.decoder3(torch.cat((x, x_res2), dim=1))
        outputs3 = self.side3(x)
        x = up_sampling(x, x_res1)  # [1, 64, 144, 144]

        x = self.decoder2(torch.cat((x, x_res1), dim=1))
        outputs2 = self.side2(x)
        x = up_sampling(x, x_res0)  # [1, 64, 288, 288]

        x = self.decoder1(torch.cat((x, x_res0), dim=1))  # [1, 64, 288, 288]
        outputs1 = self.side1(x)

        outputs2 = up_sampling(outputs2, outputs1)
        outputs3 = up_sampling(outputs3, outputs1)
        outputs4 = up_sampling(outputs4, outputs1)
        outputs5 = up_sampling(outputs5, outputs1)
        outputs6 = up_sampling(outputs6, outputs1)

        outputs = self.outputs(torch.cat((outputs1, outputs2, outputs3, outputs4, outputs5, outputs6), dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [outputs, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6]
        else:
            return torch.sigmoid(outputs)


# -----------------------------U^2Net_CBAM----------------------------- #
class U2Net_CBAM(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(U2Net_CBAM, self).__init__()

        # encoder part
        self.encoder1 = RSU7(in_ch=in_ch, mid_ch=32, out_ch=64)
        self.attention1 = CBAMBlock(channel=64)
        self.encoder2 = RSU6(in_ch=64, mid_ch=32, out_ch=128)
        self.attention2 = CBAMBlock(channel=128)
        self.encoder3 = RSU5(in_ch=128, mid_ch=64, out_ch=256)
        self.attention3 = CBAMBlock(channel=256)
        self.encoder4 = RSU4(in_ch=256, mid_ch=128, out_ch=512)
        self.attention4 = CBAMBlock(channel=512)
        self.encoder5 = RSU4F(in_ch=512, mid_ch=256, out_ch=512)
        self.attention5 = CBAMBlock(channel=512)
        self.encoder6 = RSU4F(in_ch=512, mid_ch=256, out_ch=512)
        self.attention6 = CBAMBlock(channel=512)

        # decoder part
        self.decoder5 = RSU4F(in_ch=1024, mid_ch=256, out_ch=512)
        self.decoder4 = RSU4(in_ch=1024, mid_ch=128, out_ch=256)
        self.decoder3 = RSU5(in_ch=512, mid_ch=64, out_ch=128)
        self.decoder2 = RSU6(in_ch=256, mid_ch=32, out_ch=64)
        self.decoder1 = RSU7(in_ch=128, mid_ch=16, out_ch=64)

        # side part
        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=256, out_channels=out_ch, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)

        # output part
        self.outputs = nn.Conv2d(in_channels=out_ch * 6, out_channels=out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x = self.attention1(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = down_sampling(x)
        x = self.encoder2(x)
        x = self.attention2(x)
        x_res1 = x  # [1, 128, 144, 144]

        x = down_sampling(x)
        x = self.encoder3(x)
        x = self.attention3(x)
        x_res2 = x  # [1, 256, 72, 72]

        x = down_sampling(x)
        x = self.encoder4(x)
        x = self.attention4(x)
        x_res3 = x  # [1, 512, 36, 36]

        x = down_sampling(x)
        x = self.encoder5(x)
        x = self.attention5(x)
        x_res4 = x  # [1, 512, 18, 18]

        x = down_sampling(x)
        x = self.encoder6(x)
        x = self.attention6(x)  # [1, 512, 9, 9]

        x = up_sampling(x, x_res4)  # [1, 512, 18, 18]
        outputs6 = self.side6(x)

        x = self.decoder5(torch.cat((x, x_res4), dim=1))
        outputs5 = self.side5(x)
        x = up_sampling(x, x_res3)  # [1, 512, 36, 36]

        x = self.decoder4(torch.cat((x, x_res3), dim=1))
        outputs4 = self.side4(x)
        x = up_sampling(x, x_res2)  # [1, 256, 72, 72]

        x = self.decoder3(torch.cat((x, x_res2), dim=1))
        outputs3 = self.side3(x)
        x = up_sampling(x, x_res1)  # [1, 128, 144, 144]

        x = self.decoder2(torch.cat((x, x_res1), dim=1))
        outputs2 = self.side2(x)
        x = up_sampling(x, x_res0)  # [1, 64, 288, 288]

        x = self.decoder1(torch.cat((x, x_res0), dim=1))  # [1, 64, 288, 288]
        outputs1 = self.side1(x)

        outputs2 = up_sampling(outputs2, outputs1)
        outputs3 = up_sampling(outputs3, outputs1)
        outputs4 = up_sampling(outputs4, outputs1)
        outputs5 = up_sampling(outputs5, outputs1)
        outputs6 = up_sampling(outputs6, outputs1)

        outputs = self.outputs(torch.cat((outputs1, outputs2, outputs3, outputs4, outputs5, outputs6), dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [outputs, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6]
        else:
            return torch.sigmoid(outputs)


# -----------------------------U^2Net_lite----------------------------- #
class U2Net_lite_CBAM(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(U2Net_lite_CBAM, self).__init__()

        # encoder part
        self.encoder1 = RSU7(in_ch=in_ch, mid_ch=16, out_ch=64)
        self.attention1 = CBAMBlock(channel=64)
        self.encoder2 = RSU6(in_ch=64, mid_ch=16, out_ch=64)
        self.attention2 = CBAMBlock(channel=64)
        self.encoder3 = RSU5(in_ch=64, mid_ch=16, out_ch=64)
        self.attention3 = CBAMBlock(channel=64)
        self.encoder4 = RSU4(in_ch=64, mid_ch=16, out_ch=64)
        self.attention4 = CBAMBlock(channel=64)
        self.encoder5 = RSU4F(in_ch=64, mid_ch=16, out_ch=64)
        self.attention5 = CBAMBlock(channel=64)
        self.encoder6 = RSU4F(in_ch=64, mid_ch=16, out_ch=64)
        self.attention6 = CBAMBlock(channel=64)

        # decoder part
        self.decoder5 = RSU4F(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder4 = RSU4(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder3 = RSU5(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder2 = RSU6(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder1 = RSU7(in_ch=128, mid_ch=16, out_ch=64)

        # side part
        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)

        # output part
        self.outputs = nn.Conv2d(in_channels=out_ch * 6, out_channels=out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x = self.attention1(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = down_sampling(x)
        x = self.encoder2(x)
        x = self.attention2(x)
        x_res1 = x  # [1, 64, 144, 144]

        x = down_sampling(x)
        x = self.encoder3(x)
        x = self.attention3(x)
        x_res2 = x  # [1, 64, 72, 72]

        x = down_sampling(x)
        x = self.encoder4(x)
        x = self.attention4(x)
        x_res3 = x  # [1, 64, 36, 36]

        x = down_sampling(x)
        x = self.encoder5(x)
        x = self.attention5(x)
        x_res4 = x  # [1, 64, 18, 18]

        x = down_sampling(x)
        x = self.encoder6(x)
        x = self.attention6(x)  # [1, 64, 9, 9]

        x = up_sampling(x, x_res4)  # [1, 64, 18, 18]
        outputs6 = self.side6(x)

        x = self.decoder5(torch.cat((x, x_res4), dim=1))
        outputs5 = self.side5(x)
        x = up_sampling(x, x_res3)  # [1, 64, 36, 36]

        x = self.decoder4(torch.cat((x, x_res3), dim=1))
        outputs4 = self.side4(x)
        x = up_sampling(x, x_res2)  # [1, 64, 72, 72]

        x = self.decoder3(torch.cat((x, x_res2), dim=1))
        outputs3 = self.side3(x)
        x = up_sampling(x, x_res1)  # [1, 64, 144, 144]

        x = self.decoder2(torch.cat((x, x_res1), dim=1))
        outputs2 = self.side2(x)
        x = up_sampling(x, x_res0)  # [1, 64, 288, 288]

        x = self.decoder1(torch.cat((x, x_res0), dim=1))  # [1, 64, 288, 288]
        outputs1 = self.side1(x)

        outputs2 = up_sampling(outputs2, outputs1)
        outputs3 = up_sampling(outputs3, outputs1)
        outputs4 = up_sampling(outputs4, outputs1)
        outputs5 = up_sampling(outputs5, outputs1)
        outputs6 = up_sampling(outputs6, outputs1)

        outputs = self.outputs(torch.cat((outputs1, outputs2, outputs3, outputs4, outputs5, outputs6), dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [outputs, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6]
        else:
            return torch.sigmoid(outputs)


# -----------------------------U^2Net_DW----------------------------- #
class U2Net_DW(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(U2Net_DW, self).__init__()

        # encoder part
        self.encoder1 = DWRSU7(in_ch=in_ch, mid_ch=32, out_ch=64)
        self.encoder2 = DWRSU6(in_ch=64, mid_ch=32, out_ch=128)
        self.encoder3 = DWRSU5(in_ch=128, mid_ch=64, out_ch=256)
        self.encoder4 = DWRSU4(in_ch=256, mid_ch=128, out_ch=512)
        self.encoder5 = DWRSU4F(in_ch=512, mid_ch=256, out_ch=512)
        self.encoder6 = DWRSU4F(in_ch=512, mid_ch=256, out_ch=512)

        # decoder part
        self.decoder5 = DWRSU4F(in_ch=1024, mid_ch=256, out_ch=512)
        self.decoder4 = DWRSU4(in_ch=1024, mid_ch=128, out_ch=256)
        self.decoder3 = DWRSU5(in_ch=512, mid_ch=64, out_ch=128)
        self.decoder2 = DWRSU6(in_ch=256, mid_ch=32, out_ch=64)
        self.decoder1 = DWRSU7(in_ch=128, mid_ch=16, out_ch=64)

        # side part
        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=256, out_channels=out_ch, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)

        # output part
        self.outputs = nn.Conv2d(in_channels=out_ch * 6, out_channels=out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = down_sampling(x)
        x = self.encoder2(x)
        x_res1 = x  # [1, 128, 144, 144]

        x = down_sampling(x)
        x = self.encoder3(x)
        x_res2 = x  # [1, 256, 72, 72]

        x = down_sampling(x)
        x = self.encoder4(x)
        x_res3 = x  # [1, 512, 36, 36]

        x = down_sampling(x)
        x = self.encoder5(x)
        x_res4 = x  # [1, 512, 18, 18]

        x = down_sampling(x)
        x = self.encoder6(x)  # [1, 512, 9, 9]

        x = up_sampling(x, x_res4)  # [1, 512, 18, 18]
        outputs6 = self.side6(x)

        x = self.decoder5(torch.cat((x, x_res4), dim=1))
        outputs5 = self.side5(x)
        x = up_sampling(x, x_res3)  # [1, 512, 36, 36]

        x = self.decoder4(torch.cat((x, x_res3), dim=1))
        outputs4 = self.side4(x)
        x = up_sampling(x, x_res2)  # [1, 256, 72, 72]

        x = self.decoder3(torch.cat((x, x_res2), dim=1))
        outputs3 = self.side3(x)
        x = up_sampling(x, x_res1)  # [1, 128, 144, 144]

        x = self.decoder2(torch.cat((x, x_res1), dim=1))
        outputs2 = self.side2(x)
        x = up_sampling(x, x_res0)  # [1, 64, 288, 288]

        x = self.decoder1(torch.cat((x, x_res0), dim=1))  # [1, 64, 288, 288]
        outputs1 = self.side1(x)

        outputs2 = up_sampling(outputs2, outputs1)
        outputs3 = up_sampling(outputs3, outputs1)
        outputs4 = up_sampling(outputs4, outputs1)
        outputs5 = up_sampling(outputs5, outputs1)
        outputs6 = up_sampling(outputs6, outputs1)

        outputs = self.outputs(torch.cat((outputs1, outputs2, outputs3, outputs4, outputs5, outputs6), dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [outputs, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6]
        else:
            return torch.sigmoid(outputs)


# -----------------------------U^2Net_lite_DW----------------------------- #
class U2Net_lite_DW(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(U2Net_lite_DW, self).__init__()

        # encoder part
        self.encoder1 = DWRSU7(in_ch=in_ch, mid_ch=16, out_ch=64)
        self.encoder2 = DWRSU6(in_ch=64, mid_ch=16, out_ch=64)
        self.encoder3 = DWRSU5(in_ch=64, mid_ch=16, out_ch=64)
        self.encoder4 = DWRSU4(in_ch=64, mid_ch=16, out_ch=64)
        self.encoder5 = DWRSU4F(in_ch=64, mid_ch=16, out_ch=64)
        self.encoder6 = DWRSU4F(in_ch=64, mid_ch=16, out_ch=64)

        # decoder part
        self.decoder5 = DWRSU4F(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder4 = DWRSU4(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder3 = DWRSU5(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder2 = DWRSU6(in_ch=128, mid_ch=16, out_ch=64)
        self.decoder1 = DWRSU7(in_ch=128, mid_ch=16, out_ch=64)

        # side part
        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)

        # output part
        self.outputs = nn.Conv2d(in_channels=out_ch * 6, out_channels=out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = down_sampling(x)
        x = self.encoder2(x)
        x_res1 = x  # [1, 64, 144, 144]

        x = down_sampling(x)
        x = self.encoder3(x)
        x_res2 = x  # [1, 64, 72, 72]

        x = down_sampling(x)
        x = self.encoder4(x)
        x_res3 = x  # [1, 64, 36, 36]

        x = down_sampling(x)
        x = self.encoder5(x)
        x_res4 = x  # [1, 64, 18, 18]

        x = down_sampling(x)
        x = self.encoder6(x)  # [1, 64, 9, 9]

        x = up_sampling(x, x_res4)  # [1, 64, 18, 18]
        outputs6 = self.side6(x)

        x = self.decoder5(torch.cat((x, x_res4), dim=1))
        outputs5 = self.side5(x)
        x = up_sampling(x, x_res3)  # [1, 64, 36, 36]

        x = self.decoder4(torch.cat((x, x_res3), dim=1))
        outputs4 = self.side4(x)
        x = up_sampling(x, x_res2)  # [1, 64, 72, 72]

        x = self.decoder3(torch.cat((x, x_res2), dim=1))
        outputs3 = self.side3(x)
        x = up_sampling(x, x_res1)  # [1, 64, 144, 144]

        x = self.decoder2(torch.cat((x, x_res1), dim=1))
        outputs2 = self.side2(x)
        x = up_sampling(x, x_res0)  # [1, 64, 288, 288]

        x = self.decoder1(torch.cat((x, x_res0), dim=1))  # [1, 64, 288, 288]
        outputs1 = self.side1(x)

        outputs2 = up_sampling(outputs2, outputs1)
        outputs3 = up_sampling(outputs3, outputs1)
        outputs4 = up_sampling(outputs4, outputs1)
        outputs5 = up_sampling(outputs5, outputs1)
        outputs6 = up_sampling(outputs6, outputs1)

        outputs = self.outputs(torch.cat((outputs1, outputs2, outputs3, outputs4, outputs5, outputs6), dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [outputs, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6]
        else:
            return torch.sigmoid(outputs)


# -----------------------------U^2Net_CBAM_DW----------------------------- #
class U2Net_CBAM_DW(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(U2Net_CBAM_DW, self).__init__()

        # encoder part
        self.encoder1 = DWRSU7(in_ch=in_ch, mid_ch=32, out_ch=64)
        self.attention1 = CBAMBlock(channel=64)
        self.encoder2 = DWRSU6(in_ch=64, mid_ch=32, out_ch=128)
        self.attention2 = CBAMBlock(channel=128)
        self.encoder3 = DWRSU5(in_ch=128, mid_ch=64, out_ch=256)
        self.attention3 = CBAMBlock(channel=256)
        self.encoder4 = DWRSU4(in_ch=256, mid_ch=128, out_ch=512)
        self.attention4 = CBAMBlock(channel=512)
        self.encoder5 = DWRSU4F(in_ch=512, mid_ch=256, out_ch=512)
        self.attention5 = CBAMBlock(channel=512)
        self.encoder6 = DWRSU4F(in_ch=512, mid_ch=256, out_ch=512)
        self.attention6 = CBAMBlock(channel=512)

        # decoder part
        self.decoder5 = DWRSU4F(in_ch=1024, mid_ch=256, out_ch=512)
        self.decoder4 = DWRSU4(in_ch=1024, mid_ch=128, out_ch=256)
        self.decoder3 = DWRSU5(in_ch=512, mid_ch=64, out_ch=128)
        self.decoder2 = DWRSU6(in_ch=256, mid_ch=32, out_ch=64)
        self.decoder1 = DWRSU7(in_ch=128, mid_ch=16, out_ch=64)

        # side part
        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=256, out_channels=out_ch, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)

        # output part
        self.outputs = nn.Conv2d(in_channels=out_ch * 6, out_channels=out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x = self.attention1(x)
        x_res0 = x  # [1, 64, 288, 288]

        x = down_sampling(x)
        x = self.encoder2(x)
        x = self.attention2(x)
        x_res1 = x  # [1, 128, 144, 144]

        x = down_sampling(x)
        x = self.encoder3(x)
        x = self.attention3(x)
        x_res2 = x  # [1, 256, 72, 72]

        x = down_sampling(x)
        x = self.encoder4(x)
        x = self.attention4(x)
        x_res3 = x  # [1, 512, 36, 36]

        x = down_sampling(x)
        x = self.encoder5(x)
        x = self.attention5(x)
        x_res4 = x  # [1, 512, 18, 18]

        x = down_sampling(x)
        x = self.encoder6(x)
        x = self.attention6(x)  # [1, 512, 9, 9]

        x = up_sampling(x, x_res4)  # [1, 512, 18, 18]
        outputs6 = self.side6(x)

        x = self.decoder5(torch.cat((x, x_res4), dim=1))
        outputs5 = self.side5(x)
        x = up_sampling(x, x_res3)  # [1, 512, 36, 36]

        x = self.decoder4(torch.cat((x, x_res3), dim=1))
        outputs4 = self.side4(x)
        x = up_sampling(x, x_res2)  # [1, 256, 72, 72]

        x = self.decoder3(torch.cat((x, x_res2), dim=1))
        outputs3 = self.side3(x)
        x = up_sampling(x, x_res1)  # [1, 128, 144, 144]

        x = self.decoder2(torch.cat((x, x_res1), dim=1))
        outputs2 = self.side2(x)
        x = up_sampling(x, x_res0)  # [1, 64, 288, 288]

        x = self.decoder1(torch.cat((x, x_res0), dim=1))  # [1, 64, 288, 288]
        outputs1 = self.side1(x)

        outputs2 = up_sampling(outputs2, outputs1)
        outputs3 = up_sampling(outputs3, outputs1)
        outputs4 = up_sampling(outputs4, outputs1)
        outputs5 = up_sampling(outputs5, outputs1)
        outputs6 = up_sampling(outputs6, outputs1)

        outputs = self.outputs(torch.cat((outputs1, outputs2, outputs3, outputs4, outputs5, outputs6), dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [outputs, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6]
        else:
            return torch.sigmoid(outputs)


# -----------------------------U-Net----------------------------- #
class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super(UNet, self).__init__()
        # encoder part
        self.encoder1 = DoubleConv(in_ch=in_ch, out_ch=64)
        self.encoder2 = DoubleConv(in_ch=64, out_ch=128)
        self.encoder3 = DoubleConv(in_ch=128, out_ch=256)
        self.encoder4 = DoubleConv(in_ch=256, out_ch=512)
        self.encoder5 = DoubleConv(512, 512, 1024)  # [in_ch, out_ch, mid_ch]

        # decoder part
        self.decoder4 = DoubleConv(1024, 256, 512)  # [in_ch, out_ch, mid_ch]
        self.decoder3 = DoubleConv(512, 128, 256)
        self.decoder2 = DoubleConv(256, 64, 128)
        self.decoder1 = DoubleConv(in_ch=128, out_ch=64)

        # side part
        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=256, out_channels=out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, padding=1)

        # output part
        self.out_conv = nn.Conv2d(in_channels=out_ch * 5, out_channels=1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x_res0 = x  # [1, 64, 288. 288]

        x = down_sampling(x)
        x = self.encoder2(x)
        x_res1 = x  # [1, 128, 144, 144]

        x = down_sampling(x)
        x = self.encoder3(x)
        x_res2 = x  # [1, 256, 72, 72]

        x = down_sampling(x)
        x = self.encoder4(x)
        x_res3 = x  # [1, 512, 36, 36]

        x = down_sampling(x)
        x = self.encoder5(x)  # [1, 512, 18, 18]

        x = up_sampling(x, x_res3)  # [1, 512, 36, 36]
        outputs4 = self.side4(x)

        x = self.decoder4(torch.cat((x, x_res3), dim=1))
        outputs3 = self.side3(x)
        x = up_sampling(x, x_res2)  # [1, 256, 72, 72]

        x = self.decoder3(torch.cat((x, x_res2), dim=1))
        outputs2 = self.side2(x)
        x = up_sampling(x, x_res1)  # [1, 128, 144, 144]

        x = self.decoder2(torch.cat((x, x_res1), dim=1))
        outputs1 = self.side1(x)
        x = up_sampling(x, x_res0)  # [1, 64, 288, 288]

        x = self.decoder1(torch.cat((x, x_res0), dim=1))
        outputs0 = self.side1(x)
        outputs1 = up_sampling(outputs1, outputs0)
        outputs2 = up_sampling(outputs2, outputs0)
        outputs3 = up_sampling(outputs3, outputs0)
        outputs4 = up_sampling(outputs4, outputs0)

        outputs = self.out_conv(torch.cat((outputs0, outputs1, outputs2, outputs3, outputs4), dim=1))
        if self.training:
            # do not use torch.sigmoid for amp safe
            return [outputs, outputs4, outputs4, outputs4, outputs4, outputs4, outputs4]
        else:
            return torch.sigmoid(outputs)


if __name__ == "__main__":
    model = U2Net_CBAM_DW()
    # summary(model.to(torch.device('cuda')), (3, 288, 288))

    inputs = torch.randn(1, 3, 288, 288)
    # flops, params = thop.profile(model, inputs=(inputs,))
    # print(f"FLOPs: {flops}  Params: {params}")

