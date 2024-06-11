import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_model import *

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class MIMO_PreEncoder(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=(1, 12), dilation=(1, 16), use_bn=False):
        super(MIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn

        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size,
                              stride=(1, 1), padding=0, dilation=dilation, bias=(not use_bn))

        self.bn = nn.BatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna / 2)

    def forward(self, x):
        width = x.shape[-1]
        x = torch.cat([x[..., -self.padding:], x, x[..., :self.padding]], axis=3)
        x = self.conv(x)
        x = x[..., int(x.shape[-1] / 2 - width / 2):int(x.shape[-1] / 2 + width / 2)]

        if self.use_bn:
            x = self.bn(x)

        return x



class SegmentationHead(nn.Module):
    def __init__(self, n_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.conv1 = conv3x3(n_channels, n_channels // 2)
        self.conv2 = conv3x3(n_channels // 2, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        print("n_channels",n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetWithMIMO(nn.Module):
    def __init__(self, n_channels=32, n_classes=1, bilinear=False,segmentation_head=True,detection_head=False):
        super(UNetWithMIMO, self).__init__()
        self.mimo_pre_encoder = MIMO_PreEncoder(in_layer=n_channels, out_layer=32)
        self.segmentation_head=segmentation_head
        # U-Net model
        self.unet = UNet(n_channels=32, n_classes=32, bilinear=bilinear)

        # Segmentation head
        self.segmentation_head = SegmentationHead(n_channels=32, out_channels=n_classes)

    def forward(self, x):
        x = self.mimo_pre_encoder(x)

        x = self.unet(x)


        out = {'Detection': [], 'Segmentation': []}

        if self.segmentation_head is not None:
            x=self.segmentation_head(x)
            Y = F.interpolate(x, (256, 224))
            out['Segmentation'] = Y


        return out
            


