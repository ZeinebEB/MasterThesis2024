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


class Detection_Header(nn.Module):

    def __init__(self, use_bn=True, reg_layer=2, input_angle_size=0):
        super(Detection_Header, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if (self.input_angle_size == 224):
            self.conv1 = conv3x3(32, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif (self.input_angle_size == 448):
            self.conv1 = conv3x3(32, 144, bias=bias, stride=(1, 2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif (self.input_angle_size == 896):
            self.conv1 = conv3x3(32, 144, bias=bias, stride=(1, 2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias, stride=(1, 2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')
            return

        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, reg_layer, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return torch.cat([cls, reg], dim=1)


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


class Segmentation_Header(nn.Module):
    def __init__(self, n_channels, out_channels):
        super(Segmentation_Header, self).__init__()
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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        detection_features = x4  # For detection
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, detection_features


class UNetWithMIMOSD(nn.Module):
    def __init__(self, n_channels=32, n_classes=1, bilinear=False, segmentation_head=True, detection_head=True):
        super(UNetWithMIMOSD, self).__init__()
        self.mimo_pre_encoder = MIMO_PreEncoder(in_layer=n_channels, out_layer=32)
        self.segmentation_head = segmentation_head
        self.detection_head = detection_head

        # Unet-Seg-seq-Net model
        self.unet = UNet(n_channels=32, n_classes=32, bilinear=bilinear)
        channels = [192, 40, 48, 56]

        # Detection head
        self.detection_header = Detection_Header(input_angle_size=channels[3] * 4, reg_layer=2)
        # Segmentation head
        self.segmentation_header = Segmentation_Header(n_channels=32, out_channels=n_classes)

    def forward(self, x):
        x = self.mimo_pre_encoder(x)
        unet_output, detection_features = self.unet(x)

        out = {'Detection': [], 'Segmentation': []}

        # Detection head
        x = self.detection_header(unet_output)
        Y = F.interpolate(x, (128, 224))
        out['Detection'] = Y

        # Segmentation head
        x = self.segmentation_header(unet_output)
        Y = F.interpolate(x, (256, 224))
        out['Segmentation'] = Y

        return out

