import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def gaussian_weight_init(m, conv_std=0.01, bn_std=0.01):

    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        m.weight.data.normal_(0, conv_std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()


def kaiming_weight_init(m, bn_std=0.02):

    classname = m.__class__.__name__
    if 'Conv3d' in classname or 'ConvTranspose3d' in classname:
        version_tokens = torch.__version__.split('.')
        if int(version_tokens[0]) == 0 and int(version_tokens[1]) < 4:
            nn.init.kaiming_normal(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, bn_std)
        m.bias.data.zero_()
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def vnet_kaiming_init(net):

    net.apply(kaiming_weight_init)


def vnet_focal_init(net, obj_p):

    net.apply(gaussian_weight_init)
    # initialize bias such as the initial predicted prob for objects are at obj_p.
    net.out_block.conv2.bias.data[1] = -np.log((1 - obj_p) / obj_p)


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv_bottle(nn.Module):
    def __init__(self, nchan, elu, act):
        super(LUConv_bottle, self).__init__()
        self.act = act
        self.relu1 = ELUCons(elu, nchan//4)
        self.conv1 = nn.Conv3d(nchan, nchan//4, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm3d(nchan//4)
        
        self.relu2 = ELUCons(elu, nchan//4)
        self.conv2 = nn.Conv3d(nchan//4, nchan//4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(nchan//4)
        
        self.relu3 = ELUCons(elu, nchan)
        self.conv3 = nn.Conv3d(nchan//4, nchan, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if self.act:
            out = self.relu3(self.bn3(self.conv3(out)))
        else:
            out = self.bn3(self.conv3(out))
        return out

class LUConv(nn.Module):
    def __init__(self, nchan, elu, act):
        super(LUConv, self).__init__()
        self.act = act
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(nchan)
        
    def forward(self, x):
        if self.act:
            out = self.relu1(self.bn1(self.conv1(x)))
        else:
            out = self.bn1(self.conv1(x))
        return out


def _make_nConv(nchan, depth, elu, use_bottle):
    layers = []
    for i in range(depth):
        if use_bottle:
            if i != depth - 1:
                layers.append(LUConv_bottle(nchan, elu, act=True))
            else:
                layers.append(LUConv_bottle(nchan, elu, act=False))
        else:
            if i != depth - 1:
                layers.append(LUConv(nchan, elu, act=True))
            else:
                layers.append(LUConv(nchan, elu, act=False))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        # x16 = torch.cat((x, x, x, x, x, x, x, x,
        #                  x, x, x, x, x, x, x, x), 1)
        # out = self.relu1(torch.add(out, x16))
        out = self.relu1(out)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False, use_bottle=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        self.ops = _make_nConv(outChans, nConvs, elu, use_bottle)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False, use_bottle=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d(p=0.2)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        self.ops = _make_nConv(outChans, nConvs, elu, use_bottle)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, outChans)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.softmax(out)
        return out


class SegmentationNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels, out_channels, elu=False):
        super(SegmentationNet, self).__init__()
        self.in_tr = InputTransition(in_channels, elu)
        self.down_tr32 = DownTransition(16, 2, elu, use_bottle=False)
        self.down_tr64 = DownTransition(32, 3, elu, use_bottle=True)
        self.down_tr128 = DownTransition(64, 4, elu, dropout=True, use_bottle=True)
        self.down_tr256 = DownTransition(128, 4, elu, dropout=True, use_bottle=True)
        self.up_tr256 = UpTransition(256, 256, 4, elu, dropout=True, use_bottle=True)
        self.up_tr128 = UpTransition(256, 128, 4, elu, dropout=True, use_bottle=True)
        self.up_tr64 = UpTransition(128, 64, 3, elu, use_bottle=False)
        self.up_tr32 = UpTransition(64, 32, 2, elu, use_bottle=False)
        self.out_tr = OutputTransition(32, out_channels, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

if __name__ == "__main__":
    import torch
    model = SegmentationNet(1, 2)
    inputs = torch.randn(64, 1, 256, 256)
    out = model(inputs)
