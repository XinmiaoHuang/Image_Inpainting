import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from utils import pconv, up, ResBlock


class hg_block(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(hg_block, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        ngf = 64
        encode = []
        filters_in = [2, 4]
        filters_out = [4, 4]
        for i in range(2):
            ngf_in = ngf * filters_in[i]
            ngf_out = ngf * filters_out[i]
            encode += [nn.Conv2d(ngf_in, ngf_out, 3, 2, 1, bias=False),
                       nn.BatchNorm2d(ngf_out),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(ngf_out, ngf_out, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(ngf_out),
                       nn.ReLU(inplace=True)]
        self.encode = nn.Sequential(*encode)

        decode = []
        filters_in = [4, 4]
        filters_out = [4, 2]
        for i in range(2):
            ngf_in = ngf * filters_in[i]
            ngf_out = ngf * filters_out[i]
            decode += [nn.Upsample(scale_factor=2),
                       nn.Conv2d(ngf_in, ngf_out, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(ngf_out),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(ngf_out, ngf_out, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(ngf_out),
                       nn.ReLU(inplace=True)]
        self.decode = nn.Sequential(*decode)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class Stage2Network(nn.Module):
    def __init__(self, input_nc):
        super(Stage2Network, self).__init__()
        self.input_nc = input_nc
        
        ngf = 64
        
        self.in_conv = nn.Sequential(*[nn.Conv2d(input_nc, ngf, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(ngf),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(ngf, ngf*2, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(ngf*2),
                                       nn.ReLU(inplace=True)])

        hg_list = []
        hg_list += [hg_block(ngf*1, ngf*2),
                    hg_block(ngf*2, ngf*4),
                    hg_block(ngf*2, ngf*4)]
        
        self.hg_blocks = nn.Sequential(*hg_list)
        
        decoder = []
        decoder += [
                    nn.Conv2d(ngf*2, ngf*2, 3, 1, 1,  bias=False),
                    nn.BatchNorm2d(ngf*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ngf*2, ngf, 3, 1, 1,  bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(inplace=True)]
        # decoder += [nn.Upsample(scale_factor=2),
        #             nn.Conv2d(ngf, ngf, 3, 1, 1,  bias=False),
        #             nn.BatchNorm2d(ngf),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(ngf, ngf, 3, 1, 1,  bias=False),
        #             nn.BatchNorm2d(ngf),
        #             nn.ReLU(inplace=True)]
        self.decoder = nn.Sequential(*decoder)

        self.out_conv = nn.Sequential(*[nn.Conv2d(ngf + 3, ngf, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(ngf),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(ngf, 3, 3, 1, 1), nn.Tanh()])
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, img, mask, x2, mask2, prev):
        masked = mask * img + (1-mask) * prev
        # fake_2x = self.upsample(masked)
        # masked_2x = x2 * mask2 + (1-mask2) * fake_2x
        x = self.in_conv(masked)
        x = self.hg_blocks(x)
        x = self.decoder(x)
        x = torch.cat((x, masked), 1)
        out = self.out_conv(x)
        return out, masked