import os
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import Model
import torchvision.utils as vutils
from Dataset import CustomDataset
from datetime import datetime
from options import BaseParser
from torch.utils.data import DataLoader
from loss import CriterionPerPixel, criterion_GAN
from stage2 import Stage2Network
from utils import set_requires_grad, normalize, de_normalize


TRAINING_PATH = "./testing_images/"
MASK_PATH = "./random_mask/"
SAVING_PATH = "./models/"


def evaluate_result(y_pred, y_true):
    img = y_pred * 255
    gt = y_true * 255

    mse = ((img - gt)**2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)

    mu1 = img.mean()
    mu2 = gt.mean()
    sigma1 = np.sqrt(((img - mu1)**2).mean())
    sigma2 = np.sqrt(((gt - mu2)**2).mean())
    sigma12 = ((img - mu1) * (gt - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    c12 = (2*sigma1*sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
    s12 = (sigma12 + C3) / (sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    
    return psnr, ssim


def test(opt):
    img_data = CustomDataset(opt.data_dir, opt.mask_dir, opt.img_size)
    custom_loader = DataLoader(img_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(opt.input_nc)
    model.init('xavier')

    Punet = model.model
    dnet = model.dnet
    Punet.to(device)
    dnet.to(device)

    if opt.checkpoint:
        log_state = torch.load(opt.checkpoint)  
        model_dict = Punet.state_dict()
        model_dict.update(log_state)
        Punet.load_state_dict(model_dict)
        print("load checkpoint.")

    print("Testing...................")

    # for counting average psnr and ssim of the testset
    # count = 0
    # total_psnr = 0
    # total_ssim = 0

    while True:
        test_batch = iter(custom_loader).next()
        for idx, item in enumerate(test_batch):
                test_batch[idx] = normalize(item)
        test_img, test_mask, test_masked, test_img_2x, test_mask_2x = test_batch
        test_img = test_img.type(torch.FloatTensor).to(device)
        test_mask = test_mask.type(torch.FloatTensor).to(device)

        test_masked = test_masked.type(torch.FloatTensor).to(device)
        test_img_2x = test_img_2x.type(torch.FloatTensor).to(device)
        test_mask_2x = test_mask_2x.type(torch.FloatTensor).to(device)
        # test output
        pred = Punet(test_masked, test_mask)
        pred = pred.detach().cpu()
        test_img = test_img.detach().cpu()
        test_masked = test_masked.detach().cpu()
        test_mask = test_mask.detach().cpu()

        pred = de_normalize(pred)
        test_img = de_normalize(test_img)
        test_masked = de_normalize(test_masked)
        test_mask = de_normalize(test_mask)

        psnr, ssim = evaluate_result(pred, test_img)
        print("PSNR: {:.5f}, SSIM: {:.5f}".format(psnr, ssim))
        
        # count += 1
        # total_psnr += psnr
        # total_ssim += ssim
        # if count > 2000:
        #     print("total_psnr: {:.5f}, total:ssim:{:.5f}".format(total_psnr / count, total_ssim / count))
        #     break

        plt.figure(figsize=(32, 16))
        plt.axis('off')
        plt.title('fake image')
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(pred[0], (1, 2, 0)))
        plt.subplot(1, 3, 2)
        plt.imshow(np.transpose(test_img[0], (1, 2, 0)))
        plt.subplot(1, 3, 3)
        plt.imshow(np.transpose(test_masked[0], (1, 2, 0)))
        plt.show()

if __name__ == '__main__':
    print('Initialized.')
    parser = BaseParser()
    opt = parser.parse()
    test(opt)
    print('Over.')
