import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def get_model(model, inputsize, numclasses, scale, init=None):
    """function to get the neural network configuration
    """
    if model=='alexnet':
        clf = AlexNet(64*scale, numclasses)
    elif model=='fc':
        clf = NeuralNetFC(input_size, scale, num_classes, 4) # the last input is the number of layers
    elif model=='VGG11' or model=='VGG13' or model=='VGG16' or model=='VGG19':
        clf = VGG(model)
    elif model=='resnet18':
        clf = ResNet18(scale, numclasses)
    elif model=='resnet34':
        clf = ResNet34(scale, numclasses)
    if init=='SN':
        clf.apply(init_weights_SN) # this is not used in the experiments of this paper
    elif init=='HN':
        clf.apply(init_weights_HN)
    
    return clf

def init_weights_SN(m):
    """ parameter initialization according to the standard normal distribution
    to use: model.apply(init_weights_SN)
    """
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0,1)
        if m.bias is not None:
            m.bias.data.normal_(0,1)
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0,1)
        if m.bias is not None:
            m.bias.data.normal_(0,1)

def init_weights_HN(m):
    """the kaiming parameter initialization with normal distribution
    to use: model.apply(init_weights_HN)
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class NeuralNetFC(nn.Module):
    """Custom module for a simple fully connected neural network classifier
       with custom depth and width
    """
    def __init__(self, input_size, hidden_units, num_classes, depth):
        super(NeuralNetFC, self).__init__()
        self.depth = depth
        self.input_size = input_size 
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.features = self._make_layers()
        
    def forward(self, x):
        
        x = x.view(-1, self.input_size)
        
        out = self.features(x)
        
        return out
    def _make_layers(self):
        layers = []
        if self.depth == 1:
            layers += [nn.Linear(self.input_size, self.num_classes)]
        else:
            layers += [nn.Linear(self.input_size, self.hidden_units),nn.ReLU(inplace=True)]
            for i in range(self.depth-1):
                layers += [nn.Linear(self.hidden_units, self.hidden_units),nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.hidden_units, self.num_classes)]
        return nn.Sequential(*layers) 
        
class AlexNet(nn.Module):
    """AlexNet configuration to be used for MNIST size images
    """
    def __init__(self, hidden_units=64, num_classes=10):
        super(AlexNet, self).__init__()
        
        self.hidden_units = hidden_units
        self.features = nn.Sequential(
            nn.Conv2d(1,hidden_units , kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden_units, 3*hidden_units, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden_units*3, hidden_units*6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units*6, hidden_units*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units*4, hidden_units*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4*hidden_units , 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4*self.hidden_units)
        x = self.classifier(x)
        return x

# VGG and ResNet configurations are for the cifar-10 like input images
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """VGG configurations for CIFAR-10 input images with custom scale 
       that scales both the number of hidden units and channels
       the default configurations are recovered with hidden_scale=1
    """
    def __init__(self, vgg_name, hidden_scale=1):
        super(VGG, self).__init__()
        self.scale = hidden_scale
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(int(512*hidden_scale), 10) # int rounds to lower integer
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out
    
    
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, int(self.scale*x), kernel_size=3, padding=1),
                           nn.BatchNorm2d(int(self.scale*x)),
                           nn.ReLU(inplace=True)]
                in_channels = int(self.scale*x)
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet configurations to be used for cifar-10/cifar-100 images with custom scale
    """
    def __init__(self, block, num_blocks, num_classes=10, scale=1):
        super(ResNet, self).__init__()
        self.in_planes = int(scale*64)

        self.conv1 = nn.Conv2d(3, int(64*scale), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*scale))
        self.layer1 = self._make_layer(block, int(64*scale), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*scale), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*scale), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*scale), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*scale)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(scale, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes = num_classes, scale=scale)

def ResNet34(scale, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes = num_classes, scale=scale)

def ResNet50(scale, num_classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes = num_classes, scale=scale)

def ResNet101(scale, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes = num_classes, scale=scale)

def ResNet152(scale, num_classes):
    return ResNet(Bottleneck, [3,8,36,3], num_classes = num_classes, scale=scale)
    
    