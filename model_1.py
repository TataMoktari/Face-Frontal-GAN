import torch.nn as nn
import torch.nn.parallel
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from model_ip.light_cnn import LightCNN_29Layers_v2
from model_ip import common

dd = pdb.set_trace

v_siz = 0
z_siz = 256 - v_siz


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


class conv_mean_pool(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(conv_mean_pool, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.pooling(out)
        return out


class mean_pool_conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(mean_pool_conv, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = x
        out = self.pooling(out)
        out = self.conv(out)
        return out


class upsample_conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(upsample_conv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)

    def forward(self, x):
        out = x
        out = self.upsample(out)
        out = self.conv(out)
        return out


class residualBlock_down(nn.Module):  # for discriminator, no batchnorm
    def __init__(self, inplanes, outplanes):
        super(residualBlock_down, self).__init__()
        self.conv_shortcut = mean_pool_conv(inplanes, outplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.conv2 = conv_mean_pool(outplanes, outplanes)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = x
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.ReLU(out)
        out = self.conv2(out)

        return shortcut + out


class residualBlock_up(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(residualBlock_up, self).__init__()
        self.conv_shortcut = upsample_conv(inplanes, outplanes)
        self.conv1 = upsample_conv(inplanes, outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = x
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.ReLU(out)
        out = self.conv2(out)

        return shortcut + out


class _G_xvz(nn.Module):
    def __init__(self):
        super(_G_xvz, self).__init__()
        # self.conv = nn.Conv2d(3, 64, 3, 1, 1) 64*64 resolution implementation
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)  # 3*128*128 --> 64*128*128
        self.resBlock0 = residualBlock_down(64, 64)  # 64*128*128 --> 64*64*64
        self.resBlock1 = residualBlock_down(64, 128)
        self.resBlock2 = residualBlock_down(128, 256)
        self.resBlock3 = residualBlock_down(256, 512)
        self.resBlock4 = residualBlock_down(512, 512)
        self.fc_v = nn.Linear(512 * 4 * 4, v_siz)
        self.fc_z = nn.Linear(512 * 4 * 4, z_siz)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.conv(x)
        out = self.resBlock0(out)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.resBlock3(out)
        out = self.resBlock4(out)
        out = out.view(-1, 512 * 4 * 4)
        v = self.fc_v(out)
        v = self.softmax(v)
        z = self.fc_z(out)

        return v, z


"Generator for training 512 Dream features to frontalized face"
class _G_vzx(nn.Module):
    def __init__(self):
        super(_G_vzx, self).__init__()
        self.fc = nn.Linear(512, 4 * 4 * 512)
        self.resBlock1 = residualBlock_up(512, 512)  # 4*4-->8*8
        self.resBlock2 = residualBlock_up(512, 256)  # 8*8-->16*16
        self.resBlock3 = residualBlock_up(256, 128)  # 16*16-->32*32
        self.resBlock4 = residualBlock_up(128, 64)  # 32*32-->64*64
        self.resBlock5 = residualBlock_up(64, 64)  # 64*64-->128*128
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.conv = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc(x)  # out: 512*4*4
        out = out.view(-1, 512, 4, 4)  # (-1, 512, 4,list_test.txt" 4)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.resBlock3(out)
        out = self.resBlock4(out)
        out = self.resBlock5(out)
        out = self.bn(out)
        out = self.ReLU(out)
        out = self.conv(out)
        out = self.tanh(out)

        return out


class _D_xvs(nn.Module):
    def __init__(self):
        super(_D_xvs, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)  # 3*64*64 --> 64*64*64
        # self.conv = nn.Conv2d(3, 64, 7, 2, 3) #3*128*128 --> 64*64*64
        self.resBlock0 = residualBlock_down(64, 64)
        self.resBlock1 = residualBlock_down(64, 128)  # 64*64*64 --> 119*32*32
        self.resBlock2 = residualBlock_down(128, 256)  # 128*32*32 --> 256*16*16
        self.resBlock3 = residualBlock_down(256, 512)  # 256*16*16 --> 512*8*8
        self.resBlock4 = residualBlock_down(512, 512)  # 512*8*8 --> 512*4*4
        # self.fc_v = nn.Linear(512*4*4, v_siz)
        self.fc_s = nn.Linear(512 * 4 * 4, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.resBlock0(x)
        x = self.resBlock1(x)  # 119*32*32
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = x.view(-1, 512 * 4 * 4)
        s = self.fc_s(x)
        return s


class IP(nn.Module):
    def __init__(self):
        super(IP, self).__init__()
        self.model_recognition = LightCNN_29Layers_v2(num_classes=243)
        self.submean = common.MeanShift(rgb_range=1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.submean(x)
        x = self.model_recognition(x)
        return x


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=243):
        super(network_29layers_v2, self).__init__()
        self.conv1 = mfm(3, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        x = F.dropout(fc, training=self.training)
        # out = self.fc2_MS(fc)
        out = self.fc2(x)
        return out, fc


