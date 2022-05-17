#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:47:42 2019

@author: sumche
"""
import math
import mlconfig
import torch
from torch import nn
import torch.nn.functional as F


params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size,
                       stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


@mlconfig.register
class EfficientNet(nn.Module):

    def __init__(self,  width_mult=1.0, depth_mult=1.0,
                 dropout_rate=0.2, num_classes=100):
        super(EfficientNet, self).__init__()
        self.width_mult = width_mult
        self.depth_mult = depth_mult
        self.dropout_rate = dropout_rate
        self.in_planes_map = {}

        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        # Layer1 outputs downsized scale 2
        out_channels = _round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]
        self.layer1 = nn.Sequential(*features)
        self.in_channels = out_channels
        self.in_planes_map[0] = self.in_channels

        # Layer 2 outputs downsized scale 4
        features = self._make_layer(settings[0])
        features += self._make_layer(settings[1])
        self.layer2 = nn.Sequential(*features)
        self.in_planes_map[1] = self.in_channels

        # Layer 3 outputs downsized scale 8
        features = self._make_layer(settings[2])
        self.layer3 = nn.Sequential(*features)
        self.in_planes_map[2] = self.in_channels

        # Layer 4 outputs downsized scale 16
        features = self._make_layer(settings[3])
        self.layer4 = nn.Sequential(*features)
        self.in_planes_map[3] = self.in_channels

        # Layer 5 outputs downsized scale 32
        features = self._make_layer(settings[4])
        features += self._make_layer(settings[5])
        self.layer5 = nn.Sequential(*features)
        self.in_planes_map[4] = self.in_channels

        # Layer 6 continues to maintain scale 32
        features = self._make_layer(settings[6])
        last_channels = _round_filters(1280, width_mult)
        features += [ConvBNReLU(self.in_channels, last_channels, 1)]
        self.last_layer = nn.Sequential(*features)
        lastconv_output_channels = last_channels
        self.eq_head = nn.Sequential(
            nn.Linear(lastconv_output_channels, lastconv_output_channels),
            nn.BatchNorm1d(lastconv_output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(lastconv_output_channels, lastconv_output_channels),
            nn.BatchNorm1d(lastconv_output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(lastconv_output_channels, 4)
        )

        self.inv_head = nn.Sequential(
            nn.Linear(lastconv_output_channels, lastconv_output_channels),
            nn.BatchNorm1d(lastconv_output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(lastconv_output_channels, lastconv_output_channels),
            nn.BatchNorm1d(lastconv_output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(lastconv_output_channels, 64)
        )
        self.classifier = nn.Sequential(
            # hidden by lijin
            # nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, setting):
        t, c, n, s, k = setting
        features = []
        out_channels = _round_filters(c, self.width_mult)
        repeats = _round_repeats(n, self.depth_mult)
        for i in range(repeats):
            stride = s if i == 0 else 1
            features += [MBConvBlock(self.in_channels, out_channels,
                                     expand_ratio=t, stride=stride,
                                     kernel_size=k)]
            self.in_channels = out_channels

        return features

    def forward(self, x ,return_features = False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # out = self.last_layer(down_sampled_32)
        # out = out.mean([2, 3])
        # out = self.classifier(out)
        if return_features:
            return self.classifier(x), self.eq_head(x), F.normalize(self.inv_head(x), dim=1)
            # return self.linear(out),  F.normalize(self.inv_head(out), dim=1), F.normalize(out, dim=1)
        else:
            return self.classifier(x)


def efficientnet_model(**kwargs):
    width_mult, depth_mult, _, dropout_rate = params['efficientnet_b2']
    model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)
    return model
