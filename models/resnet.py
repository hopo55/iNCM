from models.NCM import NearestClassMean

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import relu, avg_pool2d

# HAR
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

# HAR
class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1d, self).__init__()
        self.conv1 = conv1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.num_classes = num_classes
        self.feature_size = nf * 8 * block.expansion
        self.input_channel = 3
        
        self.conv1 = conv3x3(self.input_channel, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(self.feature_size, self.num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.classifier(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)

        return logits

class ImageNet_ResNet(nn.Module):
    def __init__(self, num_classes=10, arch="resnet18"):
        super(ImageNet_ResNet, self).__init__()

        weight = models.__dict__[arch + '_Weights'].IMAGENET1K_V1
        self.model = models.__dict__[arch](weights=weight)
        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        logits = self.model(x)

        return logits

# HAR
class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet1D, self).__init__()
        self.in_planes = nf
        self.num_classes = num_classes
        self.feature_size = nf * 8 * block.expansion
        self.input_channel = 6
        
        self.conv1 = conv1x1(self.input_channel, nf * 1)
        self.bn1 = nn.BatchNorm1d(nf * 1)

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(self.feature_size, self.num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.classifier(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)

        return logits


# Reduced ResNet18 as in GEM MIR(note that nf=20).
def ImageNet_ResNet18(out_dim=10):
    return ImageNet_ResNet(out_dim)

def Reduced_ResNet18(out_dim=10, nf=20, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim, nf, bias)

def ResNet18(out_dim=10, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim, nf, bias)

def ResNet34(out_dim=10, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], out_dim, nf, bias)

# ResNet-HAR
def HAR_ResNet18(out_dim=10, nf=64, bias=True):
    return ResNet1D(BasicBlock1d, [3, 4, 6, 3], out_dim, nf, bias)