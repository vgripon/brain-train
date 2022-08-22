import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
import random # for manifold mixup
import torchaudio.transforms as T

#### Mel Spectrogram

N_FFT = 1024
N_MELS = 64

melspec = T.MelSpectrogram(n_fft=N_FFT, n_mels=N_MELS,sample_rate = 32000)

class ConvBN1d(nn.Module):
    def __init__(self, in_f, out_f, kernel_size = 9, stride = 1, padding = 4, groups = 1, outRelu = False, leaky = False):
        super(ConvBN1d, self).__init__()
        self.conv = nn.Conv1d(in_f, out_f, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups, bias = False)
        self.bn = nn.BatchNorm1d(out_f)
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
        self.convbn1 = ConvBN1d(in_f, out_f, stride = stride, outRelu = True)
        self.convbn2 = ConvBN1d(out_f, out_f)
        self.shortcut = None if stride == 1 else ConvBN1d(in_f, out_f, kernel_size = 1, stride = stride, padding = 0)

    def forward(self, x, lbda = None, perm = None):
        y = self.convbn1(x)
        z = self.convbn2(y)
        if self.shortcut is not None:
            z += self.shortcut(x)
        else:
            z += x
        if lbda is not None:
            z = lbda * z + (1 - lbda) * z[perm]
        z = torch.relu(z)
        return z

class BottleneckBlock(nn.Module):
    def __init__(self, in_f, out_f, in_expansion = 4, stride=1):
        super(BottleneckBlock, self).__init__()
        self.convbn1 = ConvBN1d(in_expansion*in_f, out_f, kernel_size = 1, padding = 0, outRelu = True)
        self.convbn2 = ConvBN1d(out_f, out_f, stride = stride, outRelu = True)
        self.convbn3 = ConvBN1d(out_f, 4*out_f, kernel_size = 1, padding = 0)
        self.shortcut = None if stride == 1 and in_expansion == 4 else ConvBN1d(in_expansion*in_f, 4*out_f, kernel_size = 1, stride = stride, padding = 0)

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
        return torch.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, blockList, featureMaps, poolEntry=False):
        super(ResNet, self).__init__()
        self.poolEntry = poolEntry
        self.embed = ConvBN1d(1, featureMaps, outRelu = True)
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
        x = torch.nn.functional.avg_pool1d(x, 2)

        if self.poolEntry:
            pass
        mixup_layer = -1
        if mixup == "mixup":
            mixup_layer = 0
        elif mixup == "manifold mixup":
            mixup_layer = random.randint(0, len(self.blocks) + 1)
        
        if mixup_layer == 0:
            x = lbda * x + (1 - lbda) * x[perm]

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

class BasicBlockRN12(nn.Module):
    def __init__(self, in_f, out_f):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = ConvBN1d(in_f, out_f, outRelu = True, leaky = True)
        self.conv2 = ConvBN1d(out_f, out_f, outRelu = True, leaky = True)
        self.conv3 = ConvBN1d(out_f, out_f)
        self.sc = ConvBN1d(in_f, out_f, kernel_size = 1, padding = 0)

    def forward(self, x, lbda = None, perm = None):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y += self.sc(x)
        if lbda is not None:
            y = lbda * y + (1 - lbda) * y[perm]
        return torch.nn.functional.leaky_relu(y, negative_slope = 0.1)
        
class ResNet12(nn.Module):
    def __init__(self, featureMaps, poolEntry=False):
        super(ResNet12, self).__init__()
        self.poolEntry = poolEntry
        self.block1 = BasicBlockRN12(1, featureMaps)
        self.block2 = BasicBlockRN12(featureMaps, int(2.5 * featureMaps))
        self.block3 = BasicBlockRN12(int(2.5 * featureMaps), 5 * featureMaps)
        self.block4 = BasicBlockRN12(5 * featureMaps, 10 * featureMaps)
        self.mp = nn.MaxPool1d(4)

    def forward(self, x, mixup = None, lbda = None, perm = None):
        x = torch.nn.functional.avg_pool1d(x, 2)
        if self.poolEntry:
            pass
        mixup_layer = -1
        if mixup == "mixup":
            mixup_layer = 0
        elif mixup == "manifold mixup":
            mixup_layer = random.randint(0, 4)
        
        if mixup_layer == 0:
            x = lbda * x + (1 - lbda) * x[perm]

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

### model ProtNet_att adapted from https://github.com/kevinco27/attentional-similarity/blob/master/main/net/protnet_att.py
### https://arxiv.org/abs/1812.01269 
class GLU(nn.Module):
    def __init__(self, inp, out, ms=(4,4), ds=1):
        super(GLU, self).__init__()
        fs = (3,3)
        ps = (1,1)
        self.ms = ms
        self.cnn_lin = nn.Conv2d(inp, out, fs, dilation=ds, padding=ps, bias=False)
        self.bn = nn.BatchNorm2d(out)
        self.mp = nn.MaxPool2d(ms)

    def forward(self, x):
        out = F.relu(self.bn(self.cnn_lin(x)))
        return self.mp(out)


class ProtNet_att(nn.Module):
    def __init__(self,nfeat=128):
        super(ProtNet_att, self).__init__()
        self.model_name = 'ProtNet_att'

        ## Mel Spectrogram extraction and normalization 
        self.melspec = melspec
        self.normalize = nn.BatchNorm2d(1)

        ks = nfeat
        self.G1 = GLU(   1, ks*1)
        self.G2 = GLU(ks*1, ks*2)
        self.G3 = GLU(ks*2, ks*3, (1,1))
        self.att = nn.Conv2d(ks*3, ks*3, (3,3), padding=(1,1), bias=False)

    def nn_att(self, inp, att):
        att_out = att(inp)  # (130, 384, 8, 10)
        att_out = F.softmax(att_out.view(att_out.size(0), att_out.size(1), -1), dim=2)  # (130, 384, 80)

        att_sc = att_out.sum(1).view(att_out.size(0), 1, att_out.size(2))  # (130, 1, 80)
        att_sc = att_sc.div(att_out.size(1))
        att_sc = att_sc.repeat(1, att_out.size(1), 1)  # (130, 384, 80)
        return att_sc

    def forward(self, x,mixup = None,lbda = None, perm = None):
        with torch.no_grad():
            zx = self.normalize(self.melspec(x)) # (130, 1, 128, 160)
        G1 = self.G1(zx)  # (130, 128, 32, 40)
        G2 = self.G2(G1)  # (130, 256, 8, 10)
        G3 = self.G3(G2)  # (130, 384, 8, 10)

        att = self.nn_att(G3, self.att)
        embed = G3.view(G3.size(0), G3.size(1), -1) * att # (130, 384, 80)
        embed = embed.sum(-1)  # (130, 384)
        
        return embed

    def forward_protonet(self, x, xavg, xstd, n=5, m=5):
        zx = (x - xavg) / xstd  # (130, 1, 128, 160)
        G1 = self.G1(zx)  # (130, 128, 32, 40)
        G2 = self.G2(G1)  # (130, 256, 8, 10)
        G3 = self.G3(G2)  # (130, 384, 8, 10)
        att = self.nn_att(G3, self.att)
        embed = G3.view(G3.size(0), G3.size(1), -1) * att  # (130, 384, 80)
        embed = embed.sum(-1)  # (130, 384)
        ### This should correspond to (batch,embedding size) -> output for few shot 

        embed2 = embed.view(-1, n * m + 1, embed.size(1))  # (5, 26, 384)
        # query -> (5, 5, 384)
        query = embed2[:, -1].view(-1, 1, embed2.size(2)).repeat(1, n, 1)
        # support -> (5, 5, 384)
        support = embed2[:, :-1].view(embed2.size(0), n, m, -1).mean(2)
        sim = -torch.pow(query - support, 2).sum(-1)  # (5, 5)
        return sim

class CNN3(nn.Module):
    def __init__(self,nfeat=128):
        super(CNN3, self).__init__()
        self.model_name = 'CNN3'

        ## Mel Spectrogram extraction and normalization 
        self.melspec = melspec
        self.normalize = nn.BatchNorm2d(1)

        ks = nfeat
        self.G1 = GLU(   1, ks*1)
        self.G2 = GLU(ks*1, ks*2)
        self.G3 = GLU(ks*2, ks*3, (1,1))
        
    def forward(self, x,mixup = None,lbda = None, perm = None):
        with torch.no_grad():
            zx = self.normalize(self.melspec(x)) # (130, 1, 128, 160)
        G1 = self.G1(zx)  # (130, 128, 32, 40)
        G2 = self.G2(G1)  # (130, 256, 8, 10)
        G3 = self.G3(G2)  # (130, 384, 8, 10)
        return torch.mean(G3,dim=(2,3),keepdim=False)

def prepareBackbone():
    return {
        "resnet18": lambda: (ResNet(BasicBlock, [(1, 1, 1), (1, 2, 1.5), (1, 2, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 6), (1, 2, 8)], args.feature_maps), 8 * args.feature_maps),
        "resnet24": lambda: (ResNet(BasicBlock, [(1, 2, 1.23), (1, 2, 1.52), (1, 2, 1.87), (1, 2, 2.31), (1, 2, 2.85), (1, 2, 3.51), (1, 2, 4.33), (1, 2, 5.34), (1, 2, 6.59), (1, 2, 8.12), (1, 2, 10)], args.feature_maps), 10 * args.feature_maps),
        # "resnet20": lambda: (ResNet(BasicBlock, [(3, 1, 1), (3, 2, 2), (3, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        # "resnet56": lambda: (ResNet(BasicBlock, [(9, 1, 1), (9, 2, 2), (9, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        # "resnet110": lambda: (ResNet(BasicBlock, [(18, 1, 1), (18, 2, 2), (18, 2, 4)], args.feature_maps, large = large), 4 * args.feature_maps),
        # "resnet50": lambda: (ResNet(BottleneckBlock, [(3, 1, 1), (4, 2, 2), (6, 2, 4), (3, 2, 8)], args.feature_maps, large = large), 8 * 4 * args.feature_maps),
        "resnet12": lambda: (ResNet12(args.feature_maps), 10 * args.feature_maps),
        "cnn3": lambda: (CNN3(args.feature_maps), 3*args.feature_maps),
        "cnn-protnet": lambda: (ProtNet_att(args.feature_maps), 3*args.feature_maps)
        }[args.backbone.lower()]()

print(" backbones,", end='')
