import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from basenet.resnet50 import resnet50, resnet50d


def init_weights(modules):
    for m in modules:
        # print(m)
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class TraceModel(nn.Module):
    def __init__(
        self,
        output_ch=5,
        pretrained=False,
        freeze=False,
        se_module=False,
        dilated=True,
    ):
        super(TraceModel, self).__init__()

        """ Base network """
        if dilated:
            self.basenet = resnet50d(pretrained, freeze, se_module)
        else:
            self.basenet = resnet50(pretrained, freeze, se_module)

        """ U network """
        self.upconv1 = double_conv(2048, 1024, 512)
        self.upconv2 = double_conv(512, 512, 256)
        self.upconv3 = double_conv(256, 256, 64)
        self.upconv4 = double_conv(64, 64, 32)

        num_class = output_ch
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """Base network"""
        sources = self.basenet(x)

        """ U network """
        y = F.interpolate(sources[0], size=sources[1].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature
