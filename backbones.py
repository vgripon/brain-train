import torch
import torch.nn as nn
from args import args

class ConvBN2d(nn.Module):
    def __init__(self, in_f, out_f, kernel_size = 3, stride = 1, padding = 1, groups = 1, outRelu = False):
        super(ConvBN2d, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups, bias = False)
        self.bn = nn.BatchNorm2d(out_f)
        self.outRelu = outRelu

    def forward(self, x):
        y = self.bn(self.conv(x))
        if self.outRelu:
            return torch.relu(y)
        else:
            return y

class BasicBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=1, in_expansion = None):
        super(BasicBlock, self).__init__()
        self.convbn1 = ConvBN2d(in_f, out_f, stride = stride, outRelu = True)
        self.convbn2 = ConvBN2d(out_f, out_f)
        self.shortcut = None if stride == 1 else ConvBN2d(in_f, out_f, kernel_size = 1, stride = stride, padding = 0)

    def forward(self, x):
        y = self.convbn1(x)
        z = self.convbn2(y)
        if self.shortcut is not None:
            z += self.shortcut(x)
        else:
            z += x
        z = torch.relu(z)
        return z

class BottleneckBlock(nn.Module):
    def __init__(self, in_f, out_f, in_expansion = 4, stride=1):
        super(BottleneckBlock, self).__init__()
        self.convbn1 = ConvBN2d(in_expansion*in_f, out_f, kernel_size = 1, padding = 0, outRelu = True)
        self.convbn2 = ConvBN2d(out_f, out_f, stride = stride, outRelu = True)
        self.convbn3 = ConvBN2d(out_f, 4*out_f, kernel_size = 1, padding = 0)
        self.shortcut = None if stride == 1 and in_expansion == 4 else ConvBN2d(in_expansion*in_f, 4*out_f, kernel_size = 1, stride = stride, padding = 0)

    def forward(self, x):
        y = self.convbn1(x)
        z = self.convbn2(y)
        out = self.convbn3(z)
        if self.shortcut is not None:
            out += self.shortcut(x)
        else:
            out += x
        return torch.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, blockList, featureMaps, large = False):
        super(ResNet, self).__init__()
        if not large:
            self.embed = ConvBN2d(3, featureMaps, outRelu = True)
        else:
            self.embed = ConvBN2d(3, featureMaps, kernel_size=7, stride=2, padding=3, outRelu = True)
        blocks = []
        lastMult = 1
        first = True
        for (nBlocks, stride, multiplier) in blockList:
            for i in range(nBlocks):
                blocks.append(block(featureMaps * lastMult, featureMaps * multiplier, in_expansion = 1 if first else 4, stride = 1 if i > 0 else stride))
                first = False
                lastMult = multiplier
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        y = self.embed(x)
        for block in self.blocks:
            y = block(y)
        return y.mean(dim = list(range(2, len(y.shape))))

class BasicBlockRN12(nn.Module):
    def __init__(self, in_f, out_f):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = ConvBN2d(in_f, out_f, outRelu = True)
        self.conv2 = ConvBN2d(out_f, out_f, outRelu = True)
        self.conv3 = ConvBN2d(out_f, out_f)
        self.sc = ConvBN2d(in_f, out_f, kernel_size = 1, padding = 0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y += self.sc(x)
        return torch.relu(y)
        
class ResNet12(nn.Module):
    def __init__(self, featureMaps):
        super(ResNet12, self).__init__()
        self.block1 = BasicBlockRN12(3, featureMaps)
        self.block2 = BasicBlockRN12(featureMaps, int(2.5 * featureMaps))
        self.block3 = BasicBlockRN12(int(2.5 * featureMaps), 5 * featureMaps)
        self.block4 = BasicBlockRN12(5 * featureMaps, 10 * featureMaps)
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        y = self.mp(self.block1(x))
        y = self.mp(self.block2(y))
        y = self.mp(self.block3(y))
        y = self.block4(y)
        return y.mean(dim = list(range(2, len(y.shape))))

def prepareBackbone():
    large = False
    if args.backbone.lower()[-6:] == "_large":
        large = True
        args.backbone = args.backbone[:-6]
    return {
        "resnet18": lambda: (ResNet(BasicBlock, [(2, 1, 1), (2, 2, 2), (2, 2, 4), (2, 2, 8)], args.feature_maps, large = large), 8 * args.feature_maps),
        "resnet20": lambda: (ResNet(BasicBlock, [(3, 1, 1), (3, 2, 2), (3, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        "resnet56": lambda: (ResNet(BasicBlock, [(9, 1, 1), (9, 2, 2), (9, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        "resnet110": lambda: (ResNet(BasicBlock, [(18, 1, 1), (18, 2, 2), (18, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        "resnet50": lambda: (ResNet(BottleneckBlock, [(3, 1, 1), (4, 2, 2), (6, 2, 4), (3, 2, 8)], args.feature_maps, large = large), 8 * 4 * args.feature_maps),
        "resnet12": lambda: (ResNet12(args.feature_maps), 10 * args.feature_maps)
        }[args.backbone.lower()]()

print(" backbones,", end='')
