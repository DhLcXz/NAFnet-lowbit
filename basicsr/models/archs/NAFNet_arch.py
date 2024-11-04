# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import sys
sys.path.append('../../../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
# from torchinfo import summary
import numpy as np
from scipy.linalg import hadamard
# from hadamard_transform import hadamard_transform

def find_min_power(x, p=2):
    y = 1
    while y<x:
        y *= p
    return y


class SoftThresholding(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.T = torch.nn.Parameter(torch.rand(self.num_features)/10)
              
    def forward(self, x):
#         print(x.shape,self.T.shape)
#         return torch.mul(torch.sign(x), torch.nn.functional.relu(torch.abs(x)-torch.abs(self.T)))
        return torch.mul(torch.tanh(x), torch.nn.functional.relu(torch.abs(x)-torch.abs(self.T)))


def hadamard_transform(u, axis=-1, fast=True):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """  
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    if fast:
        x = u[..., np.newaxis]
        for d in range(m)[::-1]:
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
        y = x.squeeze(-2) / 2**(m / 2)
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H.t()/np.sqrt(n)
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y


# +
class WHT_expansion(torch.nn.Module):
    """
    通道数扩展
    num_features: Length of the last axis, should be interger power of 2. If not, we pad 0s.
    residual: Apply shortcut connection or not
    retain_DC: Retain DC channel (the first channel) or not
    """
    def __init__(self, input_features , output_features , residual=False , retain_DC=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.num_features_pad = find_min_power(self.output_features)  
        self.ST = SoftThresholding(self.num_features_pad)    
        self.residual = residual
        self.retain_DC = retain_DC

         
    def forward(self, x):
        input_features = x.shape[-1]
        if input_features!= self.input_features:
            raise Exception('{}!={}'.format(input_features, self.input_features))
        if self.num_features_pad>input_features:
            f0 = torch.nn.functional.pad(x, (0, self.num_features_pad-input_features))
        else:
            f0 = x
        f1 = hadamard_transform(f0)

#         f2 = self.v*f1
        f3 = self.ST(f1)
        # 如果需要，添加直流分量
        if self.retain_DC:
            f3[..., 0] = f1[..., 0]  # 恢复直流分量
        f4 = hadamard_transform(f3)
        y = f4[..., :self.output_features]
        if self.residual:
            y = y + x
        return y


# -

class WHT_projection(torch.nn.Module):
    """
    通道数减少 
    num_features: Length of the last axis, should be interger power of 2. If not, we pad 0s.
    residual: Apply shortcut connection or not
    retain_DC: Retain DC channel (the first channel) or not
    """
    def __init__(self, input_features , output_features , residual=False , retain_DC=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.input_features_pad = find_min_power(self.input_features)  
        self.output_features_pad = find_min_power(self.output_features)  
        self.r = int(2**(self.input_features_pad - self.output_features_pad))
        self.ST = SoftThresholding(int(2**self.input_features_pad - self.r + 1))    
        
        self.residual = residual
        self.retain_DC = retain_DC
         
    def forward(self, x):
        input_features = x.shape[-1]
        if input_features!= self.input_features:
            raise Exception('{}!={}'.format(input_features, self.input_features))
        if self.input_features_pad>input_features:
            f0 = torch.nn.functional.pad(x, (0, self.input_features_pad-input_features))
        else:
            f0 = x
        f1 = hadamard_transform(f0)
            
        # 计算要平均池化的通道范围
        start_channel = 1  # 从通道1开始
        end_channel = int(2 ** self.input_features_pad - self.r + 1)  # 计算结束通道
#         print(end_channel)
        # 选择要进行平均池化的通道
        f2 = f1[:, :, :, start_channel:end_channel+1]
#         print(f2.shape)
        # 对选定通道进行平均池化
        n = f2.shape[0]
        f3 = []
        for i in range(n):
            f3.append(F.avg_pool1d(f2[i], kernel_size=self.r, stride=self.r , padding=0))
        f3 = torch.stack(f3)
        f4 = torch.cat(f1[:,:,:,0]/self.r , f3 , dim=-1)
        f5 = hadamard_transform(f4)
        y = f4[..., :self.output_features]
        if self.residual:
            y = y + x
        return y

# +
# import torch
# from torch import nn
# img=torch.arange(4*8*4*4).reshape(4,8,4,4)
# img_t = img.permute(0,3,2,1)
# # # 池化核和池化步长均为2
# # pool=nn.AvgPool1d(2,stride=2)
# # img_2=pool(img_t[0])
# # n = img_t.shape[0]
# # temp = []
# # for i in range(n):
# #     temp.append(F.avg_pool1d(img_t[i], kernel_size=2 , stride=2))
# # temp = torch.stack(temp)

# print(img,img.shape)
# print(img_t,img_t.shape)
# # print(img_2,img_2.shape)
# # print(temp , temp.shape)


# +
# net = WHT_expansion(input_features=8 , output_features=16)
# ex_ten = net(img_t)
# print(ex_ten.shape)

# +
# net1 = WHT_projection(input_features=8 , output_features=2)
# ex_ten1 = net1(img_t)
# print(ex_ten1.shape)
# -

class WHTConv2D(torch.nn.Module):
    def __init__(self, height, width, pods = 1, residual=True):
        super().__init__()
        self.height = height       
        self.width = width
        self.height_pad = find_min_power(self.height)  
        self.width_pad = find_min_power(self.width)
        self.pods = pods
        self.ST = torch.nn.ModuleList([SoftThresholding((self.height_pad, self.width_pad)) for i in range(self.pods)])
        self.v = torch.nn.ParameterList([torch.rand((self.height_pad, self.width_pad)) for i in range(self.pods)])
        self.residual = residual
        
    def forward(self, x):
        height, width = x.shape[-2:]
        if height!= self.height or width!=self.width:
            raise Exception('({}, {})!=({}, {})'.format(height, width, self.height, self.width))
     
        f0 = x
        if self.width_pad>self.width or self.height_pad>self.height:
            f0 = torch.nn.functional.pad(f0, (0, self.width_pad-self.width, 0, self.height_pad-self.height))
        
        f1 = hadamard_transform(f0, axis=-1)
        f2 = hadamard_transform(f1, axis=-2)
        
        f3 = [self.v[i]*f2 for i in range(self.pods)]
#         f4 = [self.conv[i](f3[i]) for i in range(self.pods)]
#         f5 = [self.ST[i](f4[i]) for i in range(self.pods)]
        f5 = [self.ST[i](f3[i]) for i in range(self.pods)]

        f6 = torch.stack(f5, dim=-1).sum(dim=-1)
        
        f7 = hadamard_transform(f6, axis=-1)
        f8 = hadamard_transform(f7, axis=-2)
        
        y = f8[..., :self.height, :self.width]
        
        if self.residual:
            y = y + x
        return y


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        #对conv1进行whtexpansion替换
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv1 = WHT_expansion(input_features = c , output_features = dw_channel)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        #对conv4进行whtexpansion替换
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv4 = WHT_expansion(input_features = c , output_features = ffn_channel)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = x.permute(0,3,2,1)
        x = self.conv1(x)#替换为wht
        x = x.permute(0,3,2,1)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = x.permute(0,3,2,1)
        x = self.conv4(x)#替换为wht
        x = x.permute(0,3,2,1)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[] , size=256):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

# +
# if __name__ == '__main__':
#     img_channel = 3
#     width = 32

#     # enc_blks = [2, 2, 4, 8]
#     # middle_blk_num = 12
#     # dec_blks = [2, 2, 2, 2]

#     enc_blks = [1, 1, 1, 28]
#     middle_blk_num = 1
#     dec_blks = [1, 1, 1, 1]
    
#     net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
#                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
#     print(net)
#     print(sum(p.numel() for p in net.parameters() if p.requires_grad))

#     inp_shape = (3, 256, 256)

#     from ptflops import get_model_complexity_info

#     macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

#     params = float(params[:-3])#百万
#     macs = float(macs[:-4])#百万

#     print(macs, params)

# +
# summary(net , (1,3,256,256))

# +
# x = torch.rand((10, 32, 128))
# model = WHT1D(x.shape[-1]) 
# y = model(x)

# x = torch.rand((10, 128, 32, 32))
# model2 = WHTConv2D(x.shape[-2], x.shape[-1], x.shape[-3], x.shape[-3]) 
# print(model2)
# print(sum(p.numel() for p in model2.parameters() if p.requires_grad))
# y = model2(x)

# +
# pods = 1
# conv = torch.nn.ModuleList([torch.nn.Conv2d(x.shape[-3], x.shape[-3], 1, bias=False) for i in range(pods)])
# print(conv)
# print(sum(p.numel() for p in conv.parameters() if p.requires_grad))
# -


