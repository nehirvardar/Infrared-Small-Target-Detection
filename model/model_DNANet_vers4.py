import torch
import torch.nn as nn


# Add Depthwise Seperable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out


# Replaced heavy CBAM with ultra-lightweight ECA module
class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


# Res_ECA_block (Replaces Res_CBAM_block)
class Res_ECA_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_ECA_block, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

        self.eca = ECAAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.eca(out)

        out += residual
        out = self.relu(out)
        return out


class DNANet_vers4(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter, deep_supervision=False):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2], nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1] * 2 + nb_filter[2] + nb_filter[0], nb_filter[1],
                                        num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2] * 2 + nb_filter[3] + nb_filter[1], nb_filter[2],
                                        num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1] * 3 + nb_filter[2] + nb_filter[0], nb_filter[1],
                                        num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)), self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4(self.conv0_2_1x1(x2_2)), self.up(self.conv0_1_1x1(x1_3)), x0_4], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(Final_x0_4)
            return output