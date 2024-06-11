import torch
import torch.nn as nn
import torch.nn.functional as F

from .network.deeplabv3plus import deeplabv3plus_resnet50

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class DetectionHead(nn.Module):

    def __init__(self, use_bn=True, reg_layer=2, input_angle_size=0):
        super(DetectionHead, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if (self.input_angle_size == 224):
            self.conv1 = conv3x3(1, 144, bias=bias)
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



# class SegmentationHead(nn.Module):
#     def __init__(self, n_channels, out_channels):
#         super(SegmentationHead, self).__init__()
#         self.conv1 = conv3x3(n_channels, n_channels)
#         self.conv2 = conv3x3(n_channels, out_channels)

#     def forward(self, x):

#         x = F.relu(self.conv1(x))
#         x = self.conv2(x)
#         return x



class DeepLabV3PlusWithMIMO_det(nn.Module):
    def __init__(self, n_channels=32, n_classes=1):
        super(DeepLabV3PlusWithMIMO_det, self).__init__()
        self.mimo_pre_encoder = MIMO_PreEncoder(in_layer=n_channels, out_layer=3)
        self.deeplabv3plus = deeplabv3plus_resnet50(num_classes=n_classes, output_stride=8, pretrained_backbone=True)
        channels = [192, 40, 48, 56]
        self.detection_head = DetectionHead(input_angle_size=channels[3] * 4, reg_layer=2)
        # self.segmentation_head = SegmentationHead(n_channels=1, out_channels=n_classes)

    def forward(self, x):
        out = {'Detection': [], 'Segmentation': []}

        x = self.mimo_pre_encoder(x)
        deep_output = self.deeplabv3plus(x)


        if (self.detection_head):
            x = self.detection_head(deep_output)
            Y = F.interpolate(x, (128, 224))
            out['Detection'] = Y


        # if self.segmentation_head:
        #     output = self.segmentation_head(deep_output)
        #     x = self.segmentation_head(output)
        #     Y = F.interpolate(x, (256, 224))
        #     out['Segmentation'] = Y

        return out