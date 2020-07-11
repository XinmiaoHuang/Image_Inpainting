import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision.models import vgg16
from PconvUnet import PconvUnet
from PconvUnet import Discriminator, PatchDiscriminator
import os


class Model:
    def __init__(self, input_nc):
        self.model = PconvUnet(input_nc)
        self.dnet = Discriminator(6, use_sigmoid=True)
        self.dnet_ = PatchDiscriminator(6)

    """https://github.com/vt-vl-lab/Guided-pix2pix"""
    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
        net.apply(init_func)

    def init(self, init_type='normal', init_gain=0.02):
        self.init_weights(self.model, init_type, gain=init_gain)
        self.init_weights(self.dnet, init_type, gain=init_gain)
        self.init_weights(self.dnet_, init_type, gain=init_gain)
        print('initialize network with %s' % init_type)

    def save_model(self, path):
        pass

    def load_model(self, path, load_g = True, load_d = True):
        if load_g:
            gnet_dict = torch.load(os.path.join(path, 'g_net.pth'))
            self.model.load_state_dict(gnet_dict)
        if load_d:
            dnet_dict = torch.load(os.path.join(path, 'd_net.pth'))
            self.dnet.load_state_dict(dnet_dict)



