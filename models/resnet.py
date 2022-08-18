import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        nn.init.xavier_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        nn.init.xavier_normal_(self.conv2.weight)    
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def __init__(self, block, layers, num_classes=3):
        super().__init__()
        self.cpu = torch.device('cpu')
        
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        nn.init.xavier_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.__make_layer(block, 64, layers[0])
        self.layer2 = self.__make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.__make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.__make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * 2, 512)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        self.fc2 = nn.Linear(512, num_classes)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        self.softmax = nn.Softmax(1)

    def forward(self, batches, batches_rgn, device):
        batches.to(device)
        batches_rgn.to(device)

        x = self.conv1(batches)           # 224x224
        x_rgn = self.conv1(batches_rgn)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112
        x_rgn = self.bn1(x_rgn)
        x_rgn = self.relu(x_rgn)
        x_rgn = self.maxpool(x_rgn)

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7
        x_rgn = self.layer1(x_rgn)
        x_rgn = self.layer2(x_rgn)
        x_rgn = self.layer3(x_rgn)
        x_rgn = self.layer4(x_rgn)

        x = self.avgpool(x)         # 1x1
        x_rgn = self.avgpool(x_rgn)
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        x_rgn = torch.flatten(x_rgn, 1)
        x.to(self.cpu)
        x_rgn.to(self.cpu)
        x = torch.cat([x, x_rgn], 1)
        x.to(device)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x