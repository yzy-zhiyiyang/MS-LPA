##
# ResNet18 Pretrained network to extract lip embedding
# This code is modified based on https://github.com/lordmartian/deep_avsr
##

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch



class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self, input_channels=1):
        super(ResNet, self).__init__()

        # 初始卷积层
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64,
                               kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        # self.layer5 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        return


    def forward(self, inputBatch):

        inputBatch = self.conv1(inputBatch)
        inputBatch = self.bn1(inputBatch)
        inputBatch = self.relu(inputBatch)
        inputBatch = self.maxpool(inputBatch)

        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        # batch = self.layer5(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch




