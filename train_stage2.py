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
from loss import CriterionPerPixel, CriterionD, CriterionGAN
from stage2 import Stage2Network


TRAINING_PATH = "D:/Dataset/coco/"
MASK_PATH = "./random_mask/"
SAVING_PATH = "./models/"

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train(opt):
    img_data = CustomDataset(opt.data_dir, opt.mask_dir, opt.img_size)
    custom_loader = DataLoader(img_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(opt.input_nc)
    model.init('xavier')

    Punet = model.model
    dnet = model.dnet
    refine_net = Stage2Network(opt.input_nc)

    Punet.to(device)
    dnet.to(device)
    refine_net.to(device)

    Punet.load_state_dict(torch.load("./model/recent/_gnet.pth"))

    # if opt.checkpoint:
    # refine_net.load_state_dict(torch.load('./model/_gnet.pth'))
    dnet.load_state_dict(torch.load('./model/_dnet.pth'))
    print("load checkpoint.")

    for param in Punet.parameters():
        param.requires_grad = False

    print("Training...................")

    test_img, test_mask, test_masked, test_img_2x, test_mask_2x = iter(custom_loader).next()
    test_img = test_img.type(torch.FloatTensor).to(device)
    test_mask = test_mask.type(torch.FloatTensor).to(device)
    test_masked = test_masked.type(torch.FloatTensor).to(device)
    test_img_2x = test_img_2x.type(torch.FloatTensor).to(device)
    test_mask_2x = test_mask_2x.type(torch.FloatTensor).to(device)

    # test output
    # pred, pred_2x = Punet(test_masked, test_mask)
    # pred_2x = pred_2x.detach().cpu()
    # test_img = test_img.detach().cpu()
    # test_masked = test_masked.detach().cpu()
    # plt.figure(figsize=(32, 16))
    # plt.axis('off')
    # plt.title('fake image')
    # plt.subplot(1, 3, 1)
    # plt.imshow(np.transpose(pred_2x[0], (1, 2, 0)))
    # plt.subplot(1, 3, 2)
    # plt.imshow(np.transpose(test_img[0], (1, 2, 0)))
    # plt.subplot(1, 3, 3)
    # plt.imshow(np.transpose(test_masked[0], (1, 2, 0)))
    # plt.show()

    optimizer = torch.optim.Adam(refine_net.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(dnet.parameters(), lr=2 * opt.lr)

    criterion = CriterionPerPixel(use_gram=True)
    criterion_D = CriterionD(False)
    criterionGAN = CriterionGAN(use_lsgan=False)

    losses = []
    losses_valid = []
    losses_hole = []
    losses_perceptual = []
    losses_tv = []
    losses_gan = []
    losses_d = []

    for epoch in range(opt.epochs):
        for idx, data in enumerate(custom_loader):
            img = data[0].type(torch.FloatTensor).to(device)
            mask = data[1].type(torch.FloatTensor).to(device)
            masked = data[2].type(torch.FloatTensor).to(device)
            img_2x = data[3].type(torch.FloatTensor).to(device)
            mask_2x = data[4].type(torch.FloatTensor).to(device)

            set_requires_grad(dnet, requires_grad=True)
            dnet.zero_grad()
            real_out, mid_real = dnet(img_2x)
            label = torch.full(real_out.size(), 1, device=device)
            real_loss = criterion_D(real_out, label)
            real_loss.backward()

            pred = Punet(masked, mask)
            pred_2x, blur_2x = refine_net(img, mask, img_2x, mask_2x, pred)

            fake_out, mid_fake = dnet(pred_2x.detach())
            label.fill_(0)
            fake_loss = criterion_D(fake_out, label)
            fake_loss.backward()

            err_d = real_loss + fake_loss
            optimizer_D.step()

            set_requires_grad(dnet, requires_grad=False)
            refine_net.zero_grad()
            fake_loss_g, mid_fake_g = dnet(pred_2x)
            label.fill_(1)
            l_gan = criterionGAN(fake_loss_g, label)
            loss_valid, loss_hole, loss_perceptual, loss_tv = criterion(pred_2x, img_2x, mask_2x)
            loss = loss_valid + 6 * loss_hole + 0.05 * loss_perceptual + \
                    0.3 * loss_tv  + 2.5 * l_gan
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            losses_valid.append(loss_valid.item())
            losses_hole.append(loss_hole.item())
            losses_perceptual.append(loss_perceptual.item())
            losses_tv.append(loss_tv.item())
            losses_gan.append(l_gan.item())
            losses_d.append(err_d.item())

            if idx % opt.save_per_iter == 0:
                time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                torch.save(refine_net.state_dict(), './model/' + '_gnet.pth')
                torch.save(dnet.state_dict(), './model/' + '_dnet.pth')
                # print('Model saved.')
            if idx % 200 == 0 or (epoch == opt.epochs -1 and idx == len(custom_loader) - 1):
                print("Epoch: {}, step: {}, loss_valid: {:.5f}, loss_hole: {:.5f}, loss_perceptual: {:.5f}, loss_total: {:.5f}, "
                     "loss_tv: {:.5f}, l_gan: {:.5f}, l_d: {:.5f}".format(epoch, idx, np.mean(losses_valid), 
                     np.mean(losses_hole), np.mean(losses_perceptual), np.mean(losses), 
                     np.mean(losses_tv), np.mean(losses_gan), np.mean(losses_d)))
                losses.clear()
                losses_valid.clear()
                losses_hole.clear()
                losses_perceptual.clear()
                losses_tv.clear()
                losses_gan.clear()
                losses_d.clear()
                with torch.no_grad():
                    pred = Punet(test_masked, test_mask)
                    pred_2x, blur_2x = refine_net(test_img, test_mask, test_img_2x, test_mask_2x, pred)
                    pred_2x = pred_2x.detach().cpu()
                    blur_2x = blur_2x.detach().cpu()
                    target = test_img_2x.detach().cpu()
                    masked = test_mask_2x.detach().cpu()
                sample_pred = vutils.make_grid(pred_2x, padding=2, normalize=False)
                sample_target = vutils.make_grid(target, padding=2, normalize=False)
                sample_masked = vutils.make_grid(masked, padding=2, normalize=False)
                sample_blur = vutils.make_grid(blur_2x, padding=2, normalize=False)
                plt.figure(figsize=(32, 16))
                plt.axis('off')
                plt.title('fake image')
                plt.subplot(4, 1, 1)
                plt.imshow(np.transpose(sample_pred, (1, 2, 0)))
                plt.subplot(4, 1, 2)
                plt.imshow(np.transpose(sample_target, (1, 2, 0)))
                plt.subplot(4, 1, 3)
                plt.imshow(np.transpose(sample_masked, (1, 2, 0)))
                plt.subplot(4, 1, 4)
                plt.imshow(np.transpose(sample_blur, (1, 2, 0)))
                plt.savefig("./sample/epoch_{}_iter_{}.png".format(epoch, idx))
                plt.close()

if __name__ == '__main__':
    print('Initialized.')
    parser = BaseParser()
    opt = parser.parse()
    train(opt)
    print('Over.')
