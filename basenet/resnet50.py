from collections import namedtuple

import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
from torchvision.models.resnet import model_urls


def init_weights(modules):
    for m in modules:
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        dilation=(1, 1),
        residual=True,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation[1],
            bias=False,
            dilation=dilation[1],
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class resnet50(torch.nn.Module):
    def __init__(self, pretrained=False, freeze=False, se_module=False):
        super(resnet50, self).__init__()
        model_urls["resnet50"] = model_urls["resnet50"].replace("https://", "http://")
        if se_module:
            resnet_pretrained_features = pretrainedmodels.se_resnet50(num_classes=1000, pretrained="imagenet")
        else:
            resnet_pretrained_features = models.resnet50(pretrained=pretrained)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        if se_module:
            self.slice1.add_module("conv1", resnet_pretrained_features.layer0.conv1)  # /2
            self.slice1.add_module("bn1", resnet_pretrained_features.layer0.bn1)
            self.slice1.add_module("relu", resnet_pretrained_features.layer0.relu1)
            self.slice2.add_module("maxpool", resnet_pretrained_features.layer0.pool)  # /2
        else:
            self.slice1.add_module("conv1", resnet_pretrained_features.conv1)  # /2
            self.slice1.add_module("bn1", resnet_pretrained_features.bn1)
            self.slice1.add_module("relu", resnet_pretrained_features.relu)
            self.slice2.add_module("maxpool", resnet_pretrained_features.maxpool)  # /2

        self.slice2.add_module("layer1", resnet_pretrained_features.layer1)
        self.slice3.add_module("layer2", resnet_pretrained_features.layer2)  # /2
        self.slice4.add_module("layer3", resnet_pretrained_features.layer3)  # /2
        self.slice5.add_module("layer4", resnet_pretrained_features.layer4)  # /2

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
            init_weights(self.slice5.modules())

        if freeze:
            # for param in self.parameters():
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h1 = h
        h = self.slice2(h)
        h2 = h
        h = self.slice3(h)
        h3 = h
        h = self.slice4(h)
        h4 = h
        h = self.slice5(h)
        h5 = h
        resnet_outputs = namedtuple("ResnetOutputs", ["h5", "h4", "h3", "h2", "h1"])
        out = resnet_outputs(h5, h4, h3, h2, h1)
        return out


class resnet50d(torch.nn.Module):
    def __init__(self, pretrained=False, freeze=False, se_module=False):
        super(resnet50d, self).__init__()
        layers = [3, 4, 6, 3]
        self.inplanes = 64

        out_dim = 512 * Bottleneck.expansion
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu1 = nn.ReLU(inplace=True)
        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer1 = self._make_layer(Bottleneck, 64, layers[0])
        layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2, dilation=2)
        layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2, dilation=2)

        # for compatibility
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        if se_module:
            self.slice1.add_module("conv1", conv1)  # /2
            self.slice1.add_module("bn1", bn1)
            self.slice1.add_module("relu", relu1)
            self.slice2.add_module("maxpool", pool)  # /2
        else:
            self.slice1.add_module("conv1", conv1)  # /2
            self.slice1.add_module("bn1", bn1)
            self.slice1.add_module("relu", relu1)
            self.slice2.add_module("maxpool", pool)  # /2

        self.slice2.add_module("layer1", layer1)
        self.slice3.add_module("layer2", layer2)  # /2
        self.slice4.add_module("layer3", layer3)  # /2
        self.slice5.add_module("layer4", layer4)  # /2

        if freeze:
            # for param in self.parameters():
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict=False)
        else:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
            init_weights(self.slice5.modules())

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.slice1(x)
        h1 = h
        h = self.slice2(h)
        h2 = h
        h = self.slice3(h)
        h3 = h
        h = self.slice4(h)
        h4 = h
        h = self.slice5(h)
        h5 = h

        resnet_outputs = namedtuple("ResnetOutputs", ["h5", "h4", "h3", "h2", "h1"])
        out = resnet_outputs(h5, h4, h3, h2, h1)

        return out


if __name__ == "__main__":
    model = resnet50(True, False, False)
    output = model(torch.randn(1, 3, 768, 768))
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)
