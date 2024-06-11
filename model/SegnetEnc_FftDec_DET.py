import torch
import torch.nn as nn
import torch.nn.functional as F

from .Segnet import SegNetEncoder

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
            self.conv1 = conv3x3(192, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif (self.input_angle_size == 448):
            self.conv1 = conv3x3(32, 144, bias=bias, stride=(1, 2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif (self.input_angle_size == 896):
            self.conv1 = conv3x3(192, 144, bias=bias, stride=(1, 2))
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

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out




class RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(RangeAngle_Decoder, self).__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0))

        self.conv_block4 = BasicBlock(96, 128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0))
        self.conv_block3 = BasicBlock(256, 192)

        self.L3 = nn.Conv2d(256, 224, kernel_size=1, stride=1, padding=0)
        self.L2 = nn.Conv2d(128, 224, kernel_size=1, stride=1, padding=0)


    def forward(self, features):
        T4 = features['x4'].transpose(1, 3)
        T3 = self.L3(features['x3']).transpose(1, 3)
        T2 = self.L2(features['x2']).transpose(1, 3)

        T4_deconv = self.deconv4(T4)

        T3_resized = F.interpolate(T3, size=(T4_deconv.size(2), T4_deconv.size(3)), mode='bilinear',
                                   align_corners=False)

        S4 = torch.cat((T4_deconv, T3_resized), dim=1)


        S4 = self.conv_block4(S4)
        S4_deconv = self.deconv3(S4)

        # print(S4_deconv.shape)  # Example output: torch.Size([batch_size, 128, height1, width1])
        # print(T2.shape)
        T2_resized = F.interpolate(T2, size=(S4_deconv.size(2), S4_deconv.size(3)), mode='bilinear',
                                   align_corners=False)

        S43 = torch.cat((S4_deconv, T2_resized), dim=1)
        # S43 = torch.cat((S4_deconv, T2), axis=1)

        out = self.conv_block3(S43)

        return out


class SegnetEnc_DET(nn.Module):
    def __init__(self, n_channels=32, BN_momentum=0.5, n_classes=3, detection_head=True):
        super(SegnetEnc_DET, self).__init__()
        self.detection_head = detection_head
        self.mimo_pre_encoder = MIMO_PreEncoder(in_layer=n_channels, out_layer=3)
        self.segnet = SegNetEncoder(in_chn=3, BN_momentum=BN_momentum)
        channels = [192, 40, 48, 56]
        self.RA_decoder = RangeAngle_Decoder()

        self.detection_head = DetectionHead(input_angle_size=channels[3] * 4, reg_layer=2)
        # self.freespace = nn.Sequential(BasicBlock(192, 128), BasicBlock(128, 64), nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, x):
        # Pass input through the U-Net encoder
        out = {'Detection': [], 'Segmentation': []}
        x = self.mimo_pre_encoder(x)

        features = self.segnet(x)
        RA = self.RA_decoder(features)

        if (self.detection_head):
            d = self.detection_head(RA)
            Y = F.interpolate(d, (128, 224))
            out['Detection'] = Y

        # if (self.segmentation_head):
        #     s = self.freespace(RA)
        #     Z = F.interpolate(s, (256, 224))
        #     out['Segmentation'] = Z

        return out