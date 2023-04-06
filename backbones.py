import torch
import torch.nn as nn
from args import args
import random # for manifold mixup
from functools import partial

class ConvBN2d(nn.Module):
    def __init__(self, in_f, out_f, kernel_size = 3, stride = 1, padding = 1, groups = 1, outRelu = False, leaky = args.leaky):
        super(ConvBN2d, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups, bias = False)
        self.bn = nn.BatchNorm2d(out_f)
        self.outRelu = outRelu
        self.leaky = leaky
        if leaky:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, lbda = None, perm = None):
        y = self.bn(self.conv(x))
        if lbda is not None:
            y = lbda * y + (1 - lbda) * y[perm]
        if self.outRelu:
            if not self.leaky:
                return torch.relu(y)
            else:
                return torch.nn.functional.leaky_relu(y, negative_slope = 0.1)
        else:
            return y

class BasicBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=1, in_expansion = None):
        super(BasicBlock, self).__init__()
        self.convbn1 = ConvBN2d(in_f, out_f, stride = stride, outRelu = True)
        self.convbn2 = ConvBN2d(out_f, out_f)
        self.shortcut = None if stride == 1 and in_f == out_f else ConvBN2d(in_f, out_f, kernel_size = 1, stride = stride, padding = 0)

    def forward(self, x, lbda = None, perm = None):
        y = self.convbn1(x)
        z = self.convbn2(y)
        if self.shortcut is not None:
            z += self.shortcut(x)
        else:
            z += x
        if lbda is not None:
            z = lbda * z + (1 - lbda) * z[perm]
        if args.leaky:
            z = torch.nn.functional.leaky_relu(z, negative_slope = 0.1)
        else:
            z = torch.relu(z)
        return z

class BasicBlock_RepVGG(nn.Module):
    def __init__(self, in_f, out_f, stride=1):
        super(BasicBlock_RepVGG, self).__init__()
        self.convbn1 = ConvBN2d(in_f, out_f, stride = stride, kernel_size = 3, padding = 1)
        self.convbn2 = ConvBN2d(in_f, out_f, stride = stride, kernel_size = 1, padding = 0)
        self.shortcut = None if stride == 1 and in_f == out_f else ConvBN2d(in_f, out_f, kernel_size = 1, stride = stride, padding = 0)
        self.bn_identity = nn.BatchNorm2d(out_f)

    def forward(self, x, lbda = None, perm = None):
        y1 = self.convbn1(x)
        y2 = self.convbn2(x)
        z = y1 + y2 
        if self.shortcut is not None:
            z += self.shortcut(x)
        else:
            z += self.bn_identity(x)

        if lbda is not None:
            z = lbda * z + (1 - lbda) * z[perm]
        if args.leaky:
            z = torch.nn.functional.leaky_relu(z, negative_slope = 0.1)
        else:
            z = torch.relu(z)

        return z

class BottleneckBlock(nn.Module):
    def __init__(self, in_f, out_f, in_expansion = 4, stride=1):
        super(BottleneckBlock, self).__init__()
        self.convbn1 = ConvBN2d(in_expansion*in_f, out_f, kernel_size = 1, padding = 0, outRelu = True)
        self.convbn2 = ConvBN2d(out_f, out_f, stride = stride, outRelu = True)
        self.convbn3 = ConvBN2d(out_f, 4*out_f, kernel_size = 1, padding = 0)
        self.shortcut = None if stride == 1 and in_expansion == 4 else ConvBN2d(in_expansion*in_f, 4*out_f, kernel_size = 1, stride = stride, padding = 0)

    def forward(self, x, lbda = None, perm = None):
        y = self.convbn1(x)
        z = self.convbn2(y)
        out = self.convbn3(z)
        if self.shortcut is not None:
            out += self.shortcut(x)
        else:
            out += x
        if lbda is not None:
            out = lbda * out + (1 - lbda) * out[perm]
        if args.leaky:
            return torch.nn.functional.leaky_relu(out, negative_slope = 0.1)
        else:
            return torch.relu(out)

class RepVGG(nn.Module):
    def __init__(self, block, blockList, a, b, featureMaps):
        super(RepVGG, self).__init__()
        features_input = min(64, int(featureMaps*a))
        self.embed = ConvBN2d(3, features_input, stride = 2)
        blocks = []
        for block_index, (nBlocks, channels) in enumerate(blockList):
            if block_index != len(blockList)-1 : 
                for i in range(nBlocks):
                    blocks.append(block(features_input, int(channels*a), stride = 1 if i > 0 else 2))
                    features_input = int(channels*a)
            else : 
                for i in range(nBlocks):
                    blocks.append(block(features_input, int(channels*b), stride = 1 if i > 0 else 2))
                    features_input = int(channels*b)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mixup = None, lbda = None, perm = None):
        mixup_layer = -1
        if mixup == "mixup":
            mixup_layer = 0
        elif mixup == "manifold mixup":
            mixup_layer = random.randint(0, len(self.blocks) + 1)
        
        if mixup_layer == 0:
            x = lbda * x + (1 - lbda) * x[perm]
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        if mixup_layer == 1:
            y = self.embed(x, lbda, perm)
        else:
            y = self.embed(x)

        for i, block in enumerate(self.blocks):
            if mixup_layer == i + 2:
                y = block(y, lbda, perm)
            else:
                y = block(y)
        y = y.mean(dim = list(range(2, len(y.shape))))
        return y

class ResNet(nn.Module):
    def __init__(self, block, blockList, featureMaps, large = False):
        super(ResNet, self).__init__()
        self.large = large
        if not large:
            self.embed = ConvBN2d(3, featureMaps, outRelu = True)
        else:
            self.embed = ConvBN2d(3, featureMaps, kernel_size=7, stride=2, padding=3, outRelu = True)
            self.mp = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        blocks = []
        lastMult = 1
        first = True
        for (nBlocks, stride, multiplier) in blockList:
            for i in range(nBlocks):
                blocks.append(block(int(featureMaps * lastMult), int(featureMaps * multiplier), in_expansion = 1 if first else 4, stride = 1 if i > 0 else stride))
                first = False
                lastMult = multiplier
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mixup = None, lbda = None, perm = None):
        mixup_layer = -1
        if mixup == "mixup":
            mixup_layer = 0
        elif mixup == "manifold mixup":
            mixup_layer = random.randint(0, len(self.blocks) + 1)
        
        if mixup_layer == 0:
            x = lbda * x + (1 - lbda) * x[perm]
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        if mixup_layer == 1:
            y = self.embed(x, lbda, perm)
        else:
            y = self.embed(x)

        if self.large:
            y = self.mp(y)

        for i, block in enumerate(self.blocks):
            if mixup_layer == i + 2:
                y = block(y, lbda, perm)
            else:
                y = block(y)

        y = y.mean(dim = list(range(2, len(y.shape))))
        return y


class BasicBlockRN12(nn.Module):
    def __init__(self, in_f, out_f):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = ConvBN2d(in_f, out_f, outRelu = True, leaky = True)
        self.conv2 = ConvBN2d(out_f, out_f, outRelu = True, leaky = True)
        self.conv3 = ConvBN2d(out_f, out_f)
        self.sc = ConvBN2d(in_f, out_f, kernel_size = 1, padding = 0)

    def forward(self, x, lbda = None, perm = None):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y += self.sc(x)
        if lbda is not None:
            y = lbda * y + (1 - lbda) * y[perm]
        if args.leaky:
            return torch.nn.functional.leaky_relu(y, negative_slope = 0.1)
        else:
            return torch.relu(y)

class ResNet12(nn.Module):
    def __init__(self, featureMaps):
        super(ResNet12, self).__init__()
        self.block1 = BasicBlockRN12(3, featureMaps)
        self.block2 = BasicBlockRN12(featureMaps, int(2.5 * featureMaps))
        self.block3 = BasicBlockRN12(int(2.5 * featureMaps), 5 * featureMaps)
        self.block4 = BasicBlockRN12(5 * featureMaps, 10 * featureMaps)
        self.mp = nn.MaxPool2d(2)

    def forward(self, x, mixup = None, lbda = None, perm = None):
        mixup_layer = -1
        if mixup == "mixup":
            mixup_layer = 0
        elif mixup == "manifold mixup":
            mixup_layer = random.randint(0, 4)
        
        if mixup_layer == 0:
            x = lbda * x + (1 - lbda) * x[perm]
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        if mixup_layer == 1:
            y = self.mp(self.block1(x, lbda, perm))
        else:
            y = self.mp(self.block1(x))

        if mixup_layer == 2:
            y = self.mp(self.block2(y, lbda, perm))
        else:
            y = self.mp(self.block2(y))

        if mixup_layer == 3:
            y = self.mp(self.block3(y, lbda, perm))
        else:
            y = self.mp(self.block3(y))

        if mixup_layer == 4:
            y = self.mp(self.block4(y, lbda, perm))
        else:
            y = self.mp(self.block4(y))
        
        y = y.mean(dim = list(range(2, len(y.shape))))
        return y
from vit import ViT
from vit_dino import vit_small, vit_base, vit_tiny

class Clip(nn.Module):
    def __init__(self, device):
        super(Clip, self).__init__()
        import clip
        self.backbone = clip.load("ViT-B/32", device=device)[0]
    def forward(self, x, mixup = None, lbda = None, perm = None):
        return self.backbone.encode_image(x)
def prepareBackbone():
    large = False
    patch_size = 0
    projection = 'conv'
    backbone = args.backbone
    if args.backbone.lower()[-6:] == "_large":
        large = True
        backbone = args.backbone[:-6]
    if 'vit' in args.backbone:
        if '_linear' in args.backbone:
            projection = 'linear'
            backbone = args.backbone[:-7]
        else:
            backbone = args.backbone
        patch_size = int(backbone.split('_')[-1])
        backbone = '_'.join(backbone.split('_')[:-1])

    return {
        "repvgg_a0": lambda: (RepVGG(BasicBlock_RepVGG, [(2, 64), (4, 128), (14, 256), (1, 512)], 0.75, 2.5, args.feature_maps), int(512 * 2.5)),
        "resnet18": lambda: (ResNet(BasicBlock, [(2, 1, 1), (2, 2, 2), (2, 2, 4), (2, 2, 8)], args.feature_maps, large = large), 8 * args.feature_maps),
        "resnet20": lambda: (ResNet(BasicBlock, [(3, 1, 1), (3, 2, 2), (3, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        "resnet56": lambda: (ResNet(BasicBlock, [(9, 1, 1), (9, 2, 2), (9, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        "resnet56flat": lambda: (ResNet(BasicBlock, [(9, 1, 1), (9, 2, 1.41), (9, 2, 2)], args.feature_maps, large = large), 2 * args.feature_maps),
        "resnet110": lambda: (ResNet(BasicBlock, [(18, 1, 1), (18, 2, 2), (18, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        "resnet50": lambda: (ResNet(BottleneckBlock, [(3, 1, 1), (4, 2, 2), (6, 2, 4), (3, 2, 8)], args.feature_maps, large = large), 8 * 4 * args.feature_maps),
        "resnet12": lambda: (ResNet12(args.feature_maps), 10 * args.feature_maps),
        "wrn28_10": lambda: (ResNet(BasicBlock, [(4, 1, 10), (4, 2, 20), (4, 2, 40)], args.feature_maps, large = large), 40 * args.feature_maps),
        "wrn16_16": lambda: (ResNet(BasicBlock, [(2, 1, 16), (2, 2, 32), (2, 2, 64)], args.feature_maps, large = large), 64 * args.feature_maps),
        "vit_tiny": lambda: (ViT(image_size = args.training_image_size, patch_size = patch_size, channels = 3, dim_head=64, dim = 192, depth = 12, heads = 3, mlp_dim = 192*4, pool=False, projection=projection, drop_path_rate=args.dropout, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)), 192),
        "vit_small": lambda: (ViT(image_size = args.training_image_size, patch_size = patch_size, channels = 3, dim_head=64, dim = 384, depth = 12, heads = 6, mlp_dim = 384*4, pool=False, projection=projection, drop_path_rate=args.dropout, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)), 384),
        "vit_base": lambda: (ViT(image_size = args.training_image_size, patch_size = patch_size, channels = 3, dim_head=64, dim = 768, depth = 12, heads = 12, mlp_dim = 768*4, pool=False, projection=projection, drop_path_rate=args.dropout, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)), 768),
        "vit_large": lambda: (ViT(image_size = args.training_image_size, patch_size = patch_size, channels = 3, dim_head=64, dim = 1024, depth = 24, heads = 16, mlp_dim = 1024*4, pool=False, projection=projection, drop_path_rate=args.dropout, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)), 1024),
        "vit_huge": lambda: (ViT(image_size = args.training_image_size, patch_size = patch_size, channels = 3, dim_head=64, dim = 1280, depth = 32, heads = 16, mlp_dim = 1280*4, pool=False, projection=projection, drop_path_rate=args.dropout, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)), 1280),
        "dino_vit_tiny": lambda: (vit_tiny(image_size = args.training_image_size, patch_size=patch_size, drop_path_rate=args.dropout), 192),
        "dino_vit_small": lambda: (vit_small(image_size = args.training_image_size, patch_size=patch_size, drop_path_rate=args.dropout), 384),
        "dino_vit_base": lambda: (vit_base(image_size = args.training_image_size, patch_size=patch_size, drop_path_rate=args.dropout), 768),
        "clip": lambda: (Clip(args.device), 512)
        }[backbone.lower()]()
print(" backbones,", end='')
