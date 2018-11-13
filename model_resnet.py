import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

import functools
from torch.autograd import Variable

dim_z = 120
vocab_size = 1000

def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SpectralNorm:
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
        fn = SpectralNorm(name)

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
    SpectralNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=2 ** 0.5):
        super().__init__()

        self.query = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

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

        self.conv0 = spectral_init(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=False if bn else True),
                                   gain=gain)
        self.conv1 = spectral_init(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=False if bn else True),
                                   gain=gain)

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = spectral_init(nn.Conv2d(in_channel, out_channel,
                                                   1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, 148)
            self.HyperBN_1 = ConditionalNorm(out_channel, 148)

    def forward(self, input, condition):
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
    def __init__(self, code_dim=100, n_class=1000):
        super().__init__()

        self.linear = nn.Linear(1000, 128, bias=False)
        # self.G_linear = spectral_init(nn.Linear(20, 4 * 4 * 1536))

        # self.conv = nn.ModuleList([GBlock(1536, 1536, n_class=n_class),
        #                            GBlock(1536, 768, n_class=n_class),
        #                            GBlock(768, 384, n_class=n_class),
        #                            GBlock(384, 192, n_class=n_class),
        #                            SelfAttention(192),
        #                            GBlock(192, 96, n_class=n_class)])
        
        self.G_linear = spectral_init(nn.Linear(20, 4 * 4 * 128))
        self.conv = nn.ModuleList([GBlock(128, 16, n_class=n_class),
                                   GBlock(16, 16, n_class=n_class),
                                   GBlock(16, 16, n_class=n_class),
                                   GBlock(16, 16, n_class=n_class),
                                   SelfAttention(16),
                                   GBlock(16, 16, n_class=n_class)])

        self.ScaledCrossReplicaBN = nn.BatchNorm2d(16)
        self.colorize = spectral_init(nn.Conv2d(16, 3, [3, 3], padding=1))

    def forward(self, input, class_id):
        codes = torch.split(input, 20, 1)
        class_emb = self.linear(class_id)  # 128

        out = self.G_linear(codes[0])
        print(out)
        # out = out.view(-1, 1536, 4, 4)
        out = out.view(-1, 128, 4, 4)
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
    def __init__(self, n_class=10):
        super().__init__()

        def conv(in_channel, out_channel, downsample=True):
            return GBlock(in_channel, out_channel,
                          bn=False,
                          upsample=False, downsample=downsample)

        gain = 2 ** 0.5

        self.pre_conv = nn.Sequential(spectral_init(nn.Conv2d(3, 64, 3,
                                                              padding=1),
                                                    gain=gain),
                                      nn.ReLU(),
                                      spectral_init(nn.Conv2d(64, 64, 3,
                                                              padding=1),
                                                    gain=gain),
                                      nn.AvgPool2d(2))
        self.pre_skip = spectral_init(nn.Conv2d(3, 64, 1))

        self.conv = nn.Sequential(conv(64, 128),
                                  conv(128, 256, downsample=False),
                                  SelfAttention(256),
                                  conv(256, 512),
                                  conv(512, 512),
                                  conv(512, 512))

        self.linear = spectral_init(nn.Linear(512, 1))

        self.embed = nn.Embedding(n_class, 512)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

    def forward(self, input, class_id):
        out = self.pre_conv(input)
        out = out + self.pre_skip(F.avg_pool2d(input, 2))

        out = self.conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        embed = self.embed(class_id)
        prod = (out * embed).sum(1)

        return out_linear + prod


from scipy.stats import truncnorm
import numpy as np

def truncated_z_sample(batch_size, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
    return truncation * values


def one_hot(index, vocab_size=vocab_size):
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output = np.zeros((num, vocab_size), dtype=np.float32)
    output[np.arange(num), index] = 1
    return output


def one_hot_if_needed(label, vocab_size=vocab_size):
    label = np.asarray(label)
    if len(label.shape) <= 1:
        label = one_hot(label, vocab_size)
    assert len(label.shape) == 2
    return label


def sample(noise, label, truncation=1., batch_size=8, vocab_size=vocab_size):
    noise = np.asarray(noise)
    label = np.asarray(label)
    num = noise.shape[0]
    if len(label.shape) == 0:
        label = np.asarray([label] * num)
    if label.shape[0] != num:
        raise ValueError('Got # noise samples ({}) != # label samples ({})'
                         .format(noise.shape[0], label.shape[0]))
    label = one_hot_if_needed(label, vocab_size)
    ims = []
    for batch_start in range(0, num, batch_size):
        s = slice(batch_start, min(num, batch_start + batch_size))
        feed_dict = {'input_z': noise[s],
                     'input_y': label[s], 'input_trunc': truncation}
    return feed_dict          
    #     ims.append(sess.run(output, feed_dict=feed_dict))
    # ims = np.concatenate(ims, axis=0)
    # assert ims.shape[0] == num
    # ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
    # ims = np.uint8(ims)
    # return ims


def interpolate(A, B, num_interps):
    alphas = np.linspace(0, 1, num_interps)
    if A.shape != B.shape:
        raise ValueError('A and B must have the same shape to interpolate.')
    return np.array([(1-a)*A + a*B for a in alphas])


def imgrid(imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
        raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows*H, cols*W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid


def imshow(a, format='png', jpeg_fallback=True):
    a = np.asarray(a, dtype=np.uint8)
    str_file = cStringIO.StringIO()
    PIL.Image.fromarray(a).save(str_file, format)
    png_data = str_file.getvalue()
    try:
        disp = IPython.display.display(IPython.display.Image(png_data))
    except IOError:
        if jpeg_fallback and format != 'jpeg':
            print('Warning: image was too large to display in format "{}"; '
                  'trying jpeg instead.').format(format)
            return imshow(a, format='jpeg')
        else:
            raise
    return disp
