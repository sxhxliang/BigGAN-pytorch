import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

import functools
from torch.autograd import Variable
from CrossReplicaBN import ScaledCrossReplicaBatchNorm2d
from spectral import SpectralNorm


class Spectral_Norm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = Spectral_Norm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    Spectral_Norm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)

def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=F.relu):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

        init_conv(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out



class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition=148):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)

        self.embed = nn.Linear(n_condition, in_channel* 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        # print(class_id.dtype)
        # print('class_id', class_id.size()) # torch.Size([4, 148])
        # print(out.size()) #torch.Size([4, 128, 4, 4])
        # class_id = torch.randn(4,1)
        # print(self.embed)
        embed = self.embed(class_id)
        # print('embed', embed.size())
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        # print(beta.size())
        out = gamma * out + beta

        return out


class GBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, n_class=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False):
        super().__init__()

        gain = 2 ** 0.5

        self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))
        self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                                   1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, 148)
            self.HyperBN_1 = ConditionalNorm(out_channel, 148)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            # print('condition',condition.size()) #condition torch.Size([4, 148])
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        if self.upsample:
            # TODO different form papers
            out = F.upsample(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip


class Generator(nn.Module):
    def __init__(self, code_dim=100, n_class=1000, chn=96, debug=False):
        super().__init__()

        self.linear = SpectralNorm(nn.Linear(n_class, 128, bias=False))
        
        if debug:
            chn = 8

        self.first_view = 16 * chn

        self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn))

        self.conv = nn.ModuleList([GBlock(16*chn, 16*chn, n_class=n_class),
                                GBlock(16*chn, 8*chn, n_class=n_class),
                                GBlock(8*chn, 4*chn, n_class=n_class),
                                GBlock(4*chn, 2*chn, n_class=n_class),
                                SelfAttention(2*chn),
                                GBlock(2*chn, 1*chn, n_class=n_class)])

        # TODO impl ScaledCrossReplicaBatchNorm 
        self.ScaledCrossReplicaBN = ScaledCrossReplicaBatchNorm2d(1*chn)
        self.colorize = SpectralNorm(nn.Conv2d(1*chn, 3, [3, 3], padding=1))

    def forward(self, input, class_id):
        codes = torch.split(input, 20, 1)
        class_emb = self.linear(class_id)  # 128

        out = self.G_linear(codes[0])
        # out = out.view(-1, 1536, 4, 4)
        out = out.view(-1, self.first_view, 4, 4)
        ids = 1
        for i, conv in enumerate(self.conv):
            if isinstance(conv, GBlock):
                
                conv_code = codes[ids]
                ids = ids+1
                condition = torch.cat([conv_code, class_emb], 1)
                # print('condition',condition.size()) #torch.Size([4, 148])
                out = conv(out, condition)

            else:
                out = conv(out)

        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)

        return F.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, n_class=1000, chn=96, debug=False):
        super().__init__()

        def conv(in_channel, out_channel, downsample=True):
            return GBlock(in_channel, out_channel,
                          bn=False,
                          upsample=False, downsample=downsample)

        gain = 2 ** 0.5
        

        if debug:
            chn = 8
        self.debug = debug

        self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(3, 1*chn, 3,padding=1),),
                                      nn.ReLU(),
                                      SpectralNorm(nn.Conv2d(1*chn, 1*chn, 3,padding=1),),
                                      nn.AvgPool2d(2))
        self.pre_skip = SpectralNorm(nn.Conv2d(3, 1*chn, 1))

        self.conv = nn.Sequential(conv(1*chn, 1*chn, downsample=True),
                                  SelfAttention(1*chn),
                                  conv(1*chn, 2*chn, downsample=True),    
                                  conv(2*chn, 4*chn, downsample=True),
                                  conv(4*chn, 8*chn, downsample=True),
                                  conv(8*chn, 16*chn, downsample=True),
                                  conv(16*chn, 16*chn, downsample=False))

        self.linear = SpectralNorm(nn.Linear(16*chn, 1))

        self.embed = nn.Embedding(n_class, 16*chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

    def forward(self, input, class_id):
        
        out = self.pre_conv(input)
        out = out + self.pre_skip(F.avg_pool2d(input, 2))
        # print(out.size())
        out = self.conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        embed = self.embed(class_id)

        prod = (out * embed).sum(1)

        # if self.debug == debug:
        #     print('class_id',class_id.size())
        #     print('out_linear',out_linear.size())
        #     print('embed', embed.size())
        #     print('prod', prod.size())

        return out_linear + prod
