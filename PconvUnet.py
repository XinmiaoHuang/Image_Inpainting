import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from utils import pconv, up, ResBlock
import functools

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(PatchDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)
        return outputs

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()
        sequence = [nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)]
        n_mult = 1
        for i in range(n_layers):
            sequence += [nn.Conv2d(ndf * n_mult, ndf * n_mult * 2, 4, 2, 1, bias=False),
                         norm_layer(ndf * n_mult * 2),
                         nn.LeakyReLU(0.2, inplace=True)]
            n_mult *= 2
            # sequence += [ResBlock(ndf * n_mult)]
        sequence += [nn.Conv2d(ndf * n_mult, 1, 4, 1, 1, bias=False)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.ModuleList(sequence)

    def addNoise(self, mu, rate=0.1):
        eps = torch.randn_like(mu)
        return mu + eps * rate

    def forward(self, input_x):   
        x = input_x
        for i in range(len(self.model)):
            x = self.model[i](x)
        return x

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetDiscriminator(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
                 padding_type='reflect', use_sigmoid=False, n_downsampling=2):
        assert (n_blocks >= 0)
        super(ResnetDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 nn.ReLU(True)]

        # n_downsampling = 2
        if n_downsampling <= 2:
            for i in range(n_downsampling):
                mult = 2 ** i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
        elif n_downsampling == 3:
            mult = 2 ** 0
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** 1
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** 2
            model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult),
                      nn.ReLU(True)]

        if n_downsampling <= 2:
            mult = 2 ** n_downsampling
        else:
            mult = 4
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        if use_sigmoid:
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
       
       
class PconvUnet(nn.Module):
    def __init__(self, input_nc):
        super(PconvUnet, self).__init__()
        self.input_nc = input_nc
        
        ngf = 64
        self.inconv = nn.Sequential(*[nn.ReflectionPad2d(3),
                       nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=7, padding=0),
                       nn.InstanceNorm2d(ngf),
                       nn.ReLU(True)])

        pconvs = [pconv(input_nc=ngf, output_nc=ngf, k_size=5, strides=2, padding=2)]
        pconvs += [pconv(input_nc=ngf, output_nc=ngf*2, k_size=5, strides=2, padding=2)]
        pconvs += [pconv(input_nc=ngf*2, output_nc=ngf*4, k_size=3, strides=2, padding=1)]
        pconvs += [pconv(input_nc=ngf*4, output_nc=ngf*8, k_size=3, strides=2, padding=1)]

        pconvs += [pconv(input_nc=ngf*8, output_nc=ngf*8, k_size=3, strides=2, padding=1)]
        pconvs += [pconv(input_nc=ngf*8, output_nc=ngf*8, k_size=3, strides=2, padding=1)]

        decoder = [up(input_nc=ngf*8, output_nc=ngf*8, k_size=3)]
        decoder += [up(input_nc=ngf*8, output_nc=ngf*4, k_size=3)]
        decoder += [up(input_nc=ngf*4, output_nc=ngf*2, k_size=3)]
        decoder += [up(input_nc=ngf*2, output_nc=ngf*1, k_size=3)]
        decoder += [up(input_nc=ngf*1, output_nc=ngf*1, k_size=3)]
        decoder += [up(input_nc=ngf*1, output_nc=ngf*1, k_size=3, cat=False)]
        
        self.pconvs = nn.ModuleList(pconvs)
        self.decoder = nn.ModuleList(decoder)

        self.out_conv = nn.Sequential(*[nn.ReflectionPad2d(3),
                                        nn.Conv2d(ngf, 3, 7, 1, 0), 
                                        nn.Tanh()])

    def addNoise(self, mu, rate=0.2):
        eps = torch.randn_like(mu)
        return mu + eps * rate

    def forward(self, x, mask):
        x = torch.cat((x, mask), dim=1)
        x_ = self.inconv(x)
        conv1, mask1 = self.pconvs[0](x_, mask)
        conv2, mask2 = self.pconvs[1](conv1, mask1)

        conv3, mask3 = self.pconvs[2](conv2, mask2)
        conv4, mask4 = self.pconvs[3](conv3, mask3)

        conv5, mask5 = self.pconvs[4](conv4, mask4)
        conv6, mask6 = self.pconvs[5](conv5, mask5)

        conv6 = self.addNoise(conv6, 0.1)
        
        up1, umask1 = self.decoder[0](conv6, mask6, conv5, mask5)
                         
        up2, umask2 = self.decoder[1](up1, umask1, conv4, mask4)
        up3, umask3 = self.decoder[2](up2, umask2, conv3, mask3)
        up4, umask4 = self.decoder[3](up3, umask3, conv2, mask2)
        up5, umask5 = self.decoder[4](up4, umask4, conv1, mask1)
        up6, umask6 = self.decoder[5](up5, umask5, x, mask)
        out = self.out_conv(up6)

        # scale 2
        # res1 = self.resnet1(up6)
        # out2 = self.out_conv2(res1)

        # # scale 4
        # res2 = self.resnet2(res1)
        # out3 = self.out_conv3(res2)

        return out