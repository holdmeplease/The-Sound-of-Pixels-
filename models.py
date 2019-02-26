import torch
import torch.nn as nn
import math
import numpy as np
import torchvision.models as v_models
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1): #3*3的二维卷积
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ModifyResNet(nn.Module):
    def __init__(self, block, layers, batch_size):
        self.inplanes = 64
        self.kchannels = 16
        self.batch_size = batch_size
        super(ModifyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # for param in self.conv1.parameters():
        #     param.requires_grad_(False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        #for param in self.parameters():
         #   param.requires_grad_(False)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # add a conv layer according to the sound of pixel
        self.myconv2 = nn.Conv2d(512, self.kchannels, kernel_size=3, padding=1)
        nn.init.constant_(self.myconv2.bias, 0.0)
        # with torch.no_grad():
        #     self.conv2.weight *= 0.0
        self.spatialmaxpool = nn.MaxPool2d(kernel_size=14)
        self.relu = nn.ReLU()

        for m in self.modules():
            #print('resnet init!')
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mode='train'):
        x = self.conv1(x)
        # print(1,x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print(2,x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(3,x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.myconv2(x)
        # temperal max pooling
        # print('myconv2', x)
        x = torch.stack([torch.max(x[3*idx:3*idx+3,:,:,:], dim=0)[0] for idx in range(self.batch_size)])
        # x = torch.max(x, dim=0, keepdim=True)[0]
        # print('x shape: ' + str(x.shape))
        # sigmoid activation
        # print(5,x)
        if mode != 'test':
            x = self.spatialmaxpool(x)
            # print('x shape: ' + str(x.shape))
        x = self.relu(x)
        return x

#初始化参数
#def weight_init(m):
#    if isinstance(m, nn.Linear):
#        nn.init.xavier_normal_(m.weight)
#        nn.init.constant_(m.bias, 0)
#    # 也可以判断是否为conv2d，使用相应的初始化方式 
#    elif isinstance(m, nn.Conv2d):
#        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#     # 是否为批归一化层
#    elif isinstance(m, nn.BatchNorm2d):
#        nn.init.constant_(m.weight, 1)
#        nn.init.constant_(m.bias, 0)

class Unet_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Unet_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True))
    def forward(self, input_Unet_block):
        output_Unet_block = self.conv(input_Unet_block)
        return output_Unet_block


channels_least = 16
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.kchannels = 16
        self.unet_block1 = Unet_block(1, channels_least)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        #left
        self.unet_block2 = Unet_block(channels_least, channels_least * 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.unet_block3 = Unet_block(channels_least * 2, channels_least * 4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.unet_block4 = Unet_block(channels_least * 4, channels_least * 8)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.unet_block5 = Unet_block(channels_least * 8, channels_least * 16)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        self.unet_block6 = Unet_block(channels_least * 16, channels_least * 32)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2)
        #bottom
        self.unet_block7 = Unet_block(channels_least * 32, channels_least * 64)
        #right
        self.t_conv6 = nn.ConvTranspose2d(channels_least * 64, channels_least * 32, 2, 2)
        self.r_unet_block6 = Unet_block(channels_least * 64, channels_least * 32)
        self.t_conv5 = nn.ConvTranspose2d(channels_least * 32, channels_least * 16, 2, 2)
        self.r_unet_block5 = Unet_block(channels_least * 32, channels_least * 16)
        self.t_conv4 = nn.ConvTranspose2d(channels_least * 16, channels_least * 8, 2, 2)
        self.r_unet_block4 = Unet_block(channels_least * 16, channels_least * 8)
        self.t_conv3 = nn.ConvTranspose2d(channels_least * 8, channels_least * 4, 2, 2)
        self.r_unet_block3 = Unet_block(channels_least * 8, channels_least * 4)
        self.t_conv2 = nn.ConvTranspose2d(channels_least * 4, channels_least * 2, 2, 2)
        self.r_unet_block2 = Unet_block(channels_least * 4, channels_least * 2)
        self.t_conv1 = nn.ConvTranspose2d(channels_least * 2, channels_least, 2, 2)
        self.r_unet_block1 = Unet_block(channels_least * 2, channels_least)
        #last conv later
        self.last_conv = nn.Conv2d(channels_least, self.kchannels, 1)
    def forward(self, inputs):
        #left
        conv1 = self.unet_block1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.unet_block2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.unet_block3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.unet_block4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        conv5 = self.unet_block5(maxpool4)
        maxpool5 = self.maxpool5(conv5)
        conv6 = self.unet_block6(maxpool5)
        maxpool6 = self.maxpool6(conv6)
        #bottom
        conv7 = self.unet_block7(maxpool6)
        #right&&skip connection
        t_conv6 = self.t_conv6(conv7)
        cat6 = torch.cat([conv6, t_conv6], 1)
        r_unet_block6 = self.r_unet_block6(cat6)
        t_conv5 = self.t_conv5(r_unet_block6)
        cat5 = torch.cat([conv5, t_conv5], 1)
        r_unet_block5 = self.r_unet_block5(cat5)
        t_conv4 = self.t_conv4(r_unet_block5)
        cat4 = torch.cat([conv4, t_conv4], 1)
        r_unet_block4 = self.r_unet_block4(cat4)
        t_conv3 = self.t_conv3(r_unet_block4)
        cat3 = torch.cat([conv3, t_conv3], 1)
        r_unet_block3 = self.r_unet_block3(cat3)
        t_conv2 = self.t_conv2(r_unet_block3)
        cat2 = torch.cat([conv2, t_conv2], 1)
        r_unet_block2 = self.r_unet_block2(cat2)
        t_conv1 = self.t_conv1(r_unet_block2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        r_unet_block1 = self.r_unet_block1(cat1)
        #不知道激活函数应该用哪个
        output_unet = F.relu(self.last_conv(r_unet_block1))
        return output_unet
    def _initialize_weights(self):
        #print('unet init!')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Synthesizer(nn.Module):
    def __init__(self):
        super(Synthesizer, self).__init__()
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
        # print(3, x)
        x = self.sigmoid(x)
        # print(4, x)
        return x
    def _initialize_weights(self):
        #print('syn init!')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


def modifyresnet18(batch_size=1):
    resnet18 = v_models.resnet18(pretrained=True)
    net = ModifyResNet(BasicBlock, [2, 2, 2, 2], batch_size)
    pretrained_dict = resnet18.state_dict()
    modified_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modified_dict}
    modified_dict.update(pretrained_dict)
    net.load_state_dict(modified_dict)
    #net.apply(weight_init)
    return net

if __name__ == '__main__':
    pass
