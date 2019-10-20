# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# Modified by Juan Perez-Rua
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseMSMetaResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseMSMetaResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # deconv layers
        self.deconv_layer_4 = self._make_deconv_layer(3, 512, [128,128,128], [4,4,4])

        # post-cnn
        self.post_cnn = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

        # reweight 
        block_meta, layers_meta = resnet_spec[18]
        reweight_layer = MetaNet(
            block_meta, layers_meta,
            feat_dim=256,
            in_channels=3,
            out_channels=self.heads['hm']+self.heads['wh'],
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.reg = nn.Conv2d(
            in_channels=256,
            out_channels=self.heads['reg'],
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.rw = reweight_layer

        self.meta_params = list(self.rw.parameters()) + \
                           list(self.reg.parameters()) + \
                           list(self.post_cnn.parameters()) + \
                           list(self.deconv_layer_4.parameters())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, inplanes, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        inplanes_ = inplanes
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes_,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            inplanes_ = planes

        return nn.Sequential(*layers)

    def forward(self, x, y):
        B = x.size(0)
        C = x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.extract_features(x)
        x = x.view(B, C, x.size(1), x.size(2), x.size(3))
        print('x feature size: ',x.shape)
        ret = {}
        rw  = self.rw(y,x)
        #print('rw size: ', rw.shape)
        ret['hm']  = rw[:,:self.heads['hm'],:,:]
        #print('hm size: ', ret['hm'].shape)
        ret['wh']  = rw[:,self.heads['hm']:self.heads['hm']+self.heads['wh'],:,:]
        #print('wh size: ', ret['wh'].shape)
        ret['reg'] = self.reg(x)

        return [ret]

    def extract_features(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.deconv_layer_4(x4)
        x = self.post_cnn(x)

        return x

    def forward_multi_class(self, x, y_codes):
        """
            x: batch of images
            y_codes: list of per-category y_code 
        """

        x = self.extract_features(x)
        ret = {}
        ret['hm'] = []
        ret['wh'] = []
        for y_code in y_codes:
            rw = self.rw.apply_code(x, y_code)
            ret['hm'].append( rw[:,:self.heads['hm'],:,:] )
            ret['wh'].append( rw[:,self.heads['hm']:self.heads['hm']+self.heads['wh'],:,:] )
        ret['hm']  = torch.cat(ret['hm'],dim=1)
        ret['wh']  = torch.cat(ret['wh'],dim=1)
        ret['reg'] = self.reg(x)

        return [ret]

    def precompute_multi_class(self, y_list):
        y_code_list = self.__getattr__('rw').extract_support_code(y_list)
        return y_code_list

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            print('BASE => init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layer_4.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.post_cnn.named_modules():
                if isinstance(m, nn.Conv2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.rw.named_modules():
                if isinstance(m, nn.Conv2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0) 
            for _, m in self.reg.named_modules():
                if isinstance(m, nn.Conv2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)                                                                           
            print('BASE => init final conv weights from normal distribution')
            for head in self.heads:
                if ('hm' in head) or ('wh' in head):
                    continue
                final_layer = self.__getattr__(head)

                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                        if m.weight.shape[0] == self.heads[head]:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('BASE => imagenet pretrained model dose not exist')
            print('BASE => please download it first')
            raise ValueError('imagenet pretrained model does not exist')

class MetaNet(nn.Module):

    """
        MetaModel to predict weights of 1D convolution
        to use with precomputed feature map.
    """
    def __init__(self, block, layers, 
                       feat_dim, in_channels, out_channels, 
                       kernel_size=1, stride=1, padding=1):

        super(MetaNet, self).__init__()
        self.inplanes = 64
        self.feat_dim = feat_dim
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=2,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)                           
        self.conv_o = nn.Conv2d(self.inplanes, 256*out_channels, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.init_weights()

        self.out_ch = out_channels

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, y, x):
        B = x.size(0)
        C = x.size(1)        
        y = y.view(-1, y.size(2), y.size(3), y.size(4), y.size(5))
        y = self.extract_support_code(y) #for batch of support sets
        y = y.view(B, C, y.size(1))
        print('RM y.size = ',y.shape)
        o = self.apply_code(x, y)        #each corresponding image x_i to y_i
        print('code size: ', o.shape)
        return o

    def apply_code(self, x, y_code):
        batch_size  = x.size(0)

        outs = []
        for xi in range(x.size(1)):
            for yi in range(y_code.size(1)):
                out = torch.nn.functional.conv2d(
                            x[:,xi,:,:,:].contiguous().view(1, batch_size*self.feat_dim, x.size(3), x.size(4)),
                            y_code[:,yi,:].contiguous().view(batch_size*self.out_ch, self.feat_dim, 1, 1), groups=batch_size,
                            bias=None
                        )
            out = out.view(batch_size, self.out_ch, outs.size(2), outs.size(3))
            outs.append(out)
        outs = torch.stack(outs)
        return outs

    def extract_support_code(self, y):
        yys = []
        for shot in range(y.size(1)):
           
            yy = self.conv1(y[:,shot,:,:,:])
            yy = self.bn1(yy)
            yy = self.relu(yy)
            yy = self.maxpool(yy)
            
            yy = self.layer1(yy)
            yy = self.layer2(yy)
            yy = self.layer3(yy)
            yy = self.layer4(yy)

            yy = self.conv_o(yy)
            yy = torch.mean(yy.view(yy.size(0), yy.size(1), -1), dim=2)
            yys.append(yy)
        y = torch.mean(torch.stack(yys), dim=0)
        return y

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            print('META => loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('META => init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)

resnet_spec = {10: (BasicBlock, [2, 2]),
               18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv, learnable):
  block_class, layers = resnet_spec[num_layers]

  model = PoseMSMetaResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers, pretrained=True)
  return model
