import torch.nn as nn


from FFTRadNet.model.Deeplab.models.aspp_module import build_aspp
from FFTRadNet.model.Deeplab.models.backbone import build_backbone
from FFTRadNet.model.Deeplab.models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepLabEncoder(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, sync_bn=True, freeze_bn=False):
        super(DeepLabEncoder, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        return x, low_level_feat

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

