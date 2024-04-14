import torch
import torch.nn as nn

from torchvision import models



def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        # In conv, the dimension of the output, if the input is H,W, is
        # H+2*padding-kernel +1
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # pooling takes Height H and width W to (H-pool)/pool+1 = H/pool, and floor. Same for W.
        # altogether, the output size is (H+2*padding-kernel +1)/pool.
    )


def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
        # input is H X W, output is   (H-1)*2 - 2*padding + kernel
    )


class AE_Net(nn.Module):

    def __init__(self, inputs=1):
        super().__init__()

        self.inputs = inputs

        if inputs <= 3:
            self.layer00 = convrelu(inputs, 6, 3, 1, 1)
            self.layer0 = convrelu(6, 40, 5, 2, 2)
        else:
            self.layer00 = convrelu(inputs, 10, 3, 1, 1)
            self.layer0 = convrelu(10, 40, 5, 2, 2)

        # def convrelu(in_channels, out_channels, kernel, padding, pool):
        self.layer1 = convrelu(40, 50, 5, 2, 2)
        self.layer10 = convrelu(50, 60, 5, 2, 1)
        self.layer2 = convrelu(60, 100, 5, 2, 2)
        self.layer20 = convrelu(100, 100, 3, 1, 1)
        self.layer3 = convrelu(100, 150, 5, 2, 2)
        self.layer4 = convrelu(150, 300, 5, 2, 2)
        self.layer5 = convrelu(300, 500, 5, 2, 2)

        self.conv_up5 = convreluT(500, 300, 4, 1)
        self.conv_up4 = convreluT(300 + 300, 150, 4, 1)
        self.conv_up3 = convreluT(150 + 150, 100, 4, 1)
        self.conv_up20 = convrelu(100 + 100, 100, 3, 1, 1)
        self.conv_up2 = convreluT(100 + 100, 60, 6, 2)
        self.conv_up10 = convrelu(60 + 60, 50, 5, 2, 1)
        self.conv_up1 = convreluT(50 + 50, 40, 6, 2)
        self.conv_up0 = convreluT(40 + 40, 20, 6, 2)
        if inputs <= 3:
            self.conv_up00 = convrelu(20 + 6 + inputs, 20, 5, 2, 1)

        else:
            self.conv_up00 = convrelu(20 + 10 + inputs, 20, 5, 2, 1)

        self.conv_up000 = convrelu(20 + inputs, 1, 5, 2, 1)

        self.Wlayer00 = convrelu(inputs + 1, 20, 3, 1, 1)
        self.Wlayer0 = convrelu(20, 30, 5, 2, 2)
        self.Wlayer1 = convrelu(30, 40, 5, 2, 2)
        self.Wlayer10 = convrelu(40, 50, 5, 2, 1)
        self.Wlayer2 = convrelu(50, 60, 5, 2, 2)
        self.Wlayer20 = convrelu(60, 70, 3, 1, 1)
        self.Wlayer3 = convrelu(70, 90, 5, 2, 2)
        self.Wlayer4 = convrelu(90, 110, 5, 2, 2)
        self.Wlayer5 = convrelu(110, 150, 5, 2, 2)

        self.Wconv_up5 = convreluT(150, 110, 4, 1)
        self.Wconv_up4 = convreluT(110 + 110, 90, 4, 1)
        self.Wconv_up3 = convreluT(90 + 90, 70, 4, 1)
        self.Wconv_up20 = convrelu(70 + 70, 60, 3, 1, 1)
        self.Wconv_up2 = convreluT(60 + 60, 50, 6, 2)
        self.Wconv_up10 = convrelu(50 + 50, 40, 5, 2, 1)
        self.Wconv_up1 = convreluT(40 + 40, 30, 6, 2)
        self.Wconv_up0 = convreluT(30 + 30, 20, 6, 2)
        self.Wconv_up00 = convrelu(20 + 20 + inputs + 1, 20, 5, 2, 1)
        self.Wconv_up000 = convrelu(20 + inputs + 1, 1, 5, 2, 1)

    def forward(self, input):

        # input0 = input[:, 0:self.inputs, :, :]
        # input0 = input.permute(0, 2,3, 1)
        # input0 = input[:, 0:self.inputs, :, :]

        input0 = input.type(torch.cuda.FloatTensor)
        layer00 = self.layer00(input0)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer2 = self.layer2(layer10)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        layer4u = self.conv_up5(layer5)
        layer4u = torch.cat([layer4u, layer4], dim=1)
        layer3u = self.conv_up4(layer4u)
        layer3u = torch.cat([layer3u, layer3], dim=1)
        layer20u = self.conv_up3(layer3u)
        layer20u = torch.cat([layer20u, layer20], dim=1)
        layer2u = self.conv_up20(layer20u)
        layer2u = torch.cat([layer2u, layer2], dim=1)
        layer10u = self.conv_up2(layer2u)
        layer10u = torch.cat([layer10u, layer10], dim=1)
        layer1u = self.conv_up10(layer10u)
        layer1u = torch.cat([layer1u, layer1], dim=1)
        layer0u = self.conv_up1(layer1u)
        layer0u = torch.cat([layer0u, layer0], dim=1)
        layer00u = self.conv_up0(layer0u)
        layer00u = torch.cat([layer00u, layer00], dim=1)
        layer00u = torch.cat([layer00u, input0], dim=1)
        layer000u = self.conv_up00(layer00u)
        layer000u = torch.cat([layer000u, input0], dim=1)
        output = self.conv_up000(layer000u)

        return output

