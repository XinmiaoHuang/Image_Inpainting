import torch
import torch.nn as nn
import torch.nn.functional as F

class pconv(nn.Module):
    def __init__(self, input_nc, output_nc, k_size, strides, padding):
        super(pconv, self).__init__()
        self.input_nc = input_nc
        conv = [nn.Conv2d(input_nc, output_nc, k_size, strides, padding),
                nn.InstanceNorm2d(output_nc),
                nn.ReLU(inplace=True)
                ]
        self.conv = nn.Sequential(*conv)
        mask_conv = [nn.Conv2d(input_nc, output_nc, k_size, strides, padding),
                    nn.Sigmoid()
                    ]
        self.mask_conv = nn.Sequential(*mask_conv)

    def forward(self, x, mask):
        mask = self.mask_conv(x)
        x = self.conv(x)
        x = (x * mask)
        return x, mask


class up(nn.Module):
    def __init__(self, input_nc, output_nc, k_size=3, strides=1, padding=1, cat=True):
        super(up, self).__init__()
        self.cat = cat
        if not cat:
            input_nc =  input_nc // 2
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv = pconv(input_nc=input_nc*2, output_nc=output_nc, k_size=k_size, strides=strides, padding=padding)
        conv = [nn.Conv2d(input_nc*2, output_nc, 3, 1, 1),
                nn.InstanceNorm2d(output_nc),
                nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*conv)
    def forward(self, x, mask, prev_x, prev_mask):
        x = self.up1(x)
        # mask = self.up2(mask)
        if self.cat:
            x = torch.cat((x, prev_x), dim=1)
        # mask = torch.cat((mask, prev_mask), dim=1)
        out = self.conv(x)
        return  out, mask


class ResBlock(nn.Module):
    def __init__(self, channels, k_size=3, stride=1, padding=1,
                 dilation=1, norm_layer=nn.BatchNorm2d, add_noise=True):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, k_size, stride, padding, dilation),
            norm_layer(channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, k_size, stride, padding, dilation),
            norm_layer(channels),
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.add_noise = add_noise

    def addNoise(self, mu, rate=0.2):
        eps = torch.randn_like(mu)
        return mu + eps * rate

    def forward(self, x):
        residual = x
        if self.add_noise:
            x = self.addNoise(x)
        x = self.conv_block1(x)
        out = self.conv_block2(x)
        out = self.relu(out + residual)
        return out

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def normalize(x):
    return 2.0 * x - 1

def de_normalize(x):
    return (x + 1.0) / 2.0

def random_crop(x, size=64, img_sz=256):
    rand_x = random.randint(0, img_sz - size)
    rand_y = random.randint(0, img_sz - size)
    return x[:, :, rand_x:rand_x+size, rand_y:rand_y+size]
