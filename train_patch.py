import os
import cv2
import torch
import torchvision
import torch.nn as nn
import random
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
from utils import normalize, de_normalize, random_crop, set_requires_grad


TRAINING_PATH = "D:/Dataset/coco/"
MASK_PATH = "./random_mask/"
SAVING_PATH = "./models/"

def train(opt):
    img_data = CustomDataset(opt.data_dir, opt.mask_dir, opt.img_size)
    custom_loader = DataLoader(img_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(opt.input_nc)
    model.init('xavier')

    Punet = model.model
    dnet = model.dnet
    pdnet = model.dnet_

    Punet.to(device)
    dnet.to(device)
    pdnet.to(device)

    if opt.checkpoint:
        log_state = torch.load(opt.checkpoint)   
        model_dict = Punet.state_dict()
        model_dict.update(log_state)
        Punet.load_state_dict(model_dict)
        dnet.load_state_dict(torch.load('./model/_dnet.pth'))
        pdnet.load_state_dict(torch.load('./model/_pdnet.pth'))
        print("load checkpoint.")

    print("Training...................")

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
    # pred = Punet(test_masked, test_mask)
    # pred = pred.detach().cpu()
    # test_img = test_img.detach().cpu()
    # test_masked = test_masked.detach().cpu()
    # plt.figure(figsize=(32, 16))
    # plt.axis('off')
    # plt.title('fake image')
    # plt.subplot(1, 3, 1)
    # plt.imshow(np.transpose(pred[0], (1, 2, 0)))
    # plt.subplot(1, 3, 2)
    # plt.imshow(np.transpose(test_img[0], (1, 2, 0)))
    # plt.subplot(1, 3, 3)
    # plt.imshow(np.transpose(test_masked[0], (1, 2, 0)))
    # plt.show()

    optimizer = torch.optim.Adam(Punet.parameters(), lr= opt.lr, betas=(0.9, 0.99))
    optimizer_D = torch.optim.Adam(dnet.parameters(), lr=opt.lr * 1.3, betas=(0.9, 0.99))
    optimizer_DP = torch.optim.Adam(pdnet.parameters(), lr=opt.lr * 1.3, betas=(0.9, 0.99))

    criterion = CriterionPerPixel(use_gram=True)
    criterion_D = criterion_GAN(use_lsgan=False)
    criterionGAN = criterion_GAN(use_lsgan=False)

    losses = []
    losses_valid = []
    losses_hole = []
    losses_perceptual = []
    losses_style = []
    losses_gan = []
    losses_d = []
    losses_dp = []
    

    for epoch in range(opt.epochs):
        for idx, data in enumerate(custom_loader):
            for i, item in enumerate(data):
                data[i] = normalize(item)
            img = data[0].type(torch.FloatTensor).to(device)
            mask = data[1].type(torch.FloatTensor).to(device)
            masked = data[2].type(torch.FloatTensor).to(device)
            # img_2x = data[3].type(torch.FloatTensor).to(device)
            # mask_2x = data[4].type(torch.FloatTensor).to(device)

            # dnet
            set_requires_grad(dnet, requires_grad=True)
            dnet.zero_grad()
            pred = Punet(masked, mask)
            fake_out = dnet(torch.cat((masked, pred.detach()), dim=1))
            real_out = dnet(torch.cat((masked, img), dim=1))

            real_loss = criterion_D(real_out, True)
            fake_loss = criterion_D(fake_out, False)

            err_d = (real_loss + fake_loss) / 2
            err_d.backward()
            optimizer_D.step()

            # patch dnet
            set_requires_grad(pdnet, requires_grad=True)
            pdnet.zero_grad()

            fake = torch.cat((img, pred.detach()), dim=1)
            real = torch.cat((img, img), dim=1)

            real_loss = 0
            fake_loss = 0

            coord = []
            rand_x = random.randint(0, 256 - 64)
            rand_y = random.randint(0, 256 - 64)
            coord.append((rand_x, rand_y))
            fake_out = fake[:, :, rand_x:rand_x+64, rand_y:rand_y+64]
            real_out = real[:, :, rand_x:rand_x+64, rand_y:rand_y+64]
            for i in range(1, 6):
                rand_x = random.randint(0, 256 - 64)
                rand_y = random.randint(0, 256 - 64)
                coord.append((rand_x, rand_y))
                fake_out = torch.cat((fake_out, fake[:, :, rand_x:rand_x+64, rand_y:rand_y+64]), dim=0)
                real_out = torch.cat((real_out, real[:, :, rand_x:rand_x+64, rand_y:rand_y+64]), dim=0)

            real_loss = criterion_D(pdnet(real_out), True)
            fake_loss = criterion_D(pdnet(fake_out), False)

            err_dp = (real_loss + fake_loss) / 2
            err_dp.backward()
            optimizer_DP.step()

            # gnet
            set_requires_grad(dnet, requires_grad=False)
            set_requires_grad(pdnet, requires_grad=False)
            Punet.zero_grad()
            fake_loss_g = dnet(torch.cat((masked, pred), dim=1))
            l_gan = criterionGAN(fake_loss_g, True)

            l_gan2 = 0
            fake_ = torch.cat((img, pred), dim=1)
            fake_set = fake_[:, :, coord[0][0]:coord[0][0]+64, coord[0][1]:coord[0][1]+64]
            for i in range(1, 6):
                rand_x = coord[i][0]
                rand_y = coord[i][1]
                fake_set = torch.cat((fake_set, fake_[:, :, rand_x:rand_x+64, rand_y:rand_y+64]), dim=0)

            fake_loss_g = pdnet(fake_set)
            l_gan2 += criterionGAN(fake_loss_g, True)

            loss_valid, loss_hole, loss_perceptual, loss_style = criterion(pred, img, mask)
            loss = loss_valid + 3 * loss_hole + 0.7 * loss_perceptual + 50 * loss_style + 0.2 * l_gan + 0.1 * l_gan2
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            losses_valid.append(loss_valid.item())
            losses_hole.append(loss_hole.item())
            losses_perceptual.append(loss_perceptual.item())
            losses_style.append(loss_style.item())
            losses_gan.append(l_gan.item())
            losses_d.append(err_d.item())
            losses_dp.append(err_dp.item())

            if idx % opt.save_per_iter == 0:
                time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                torch.save(Punet.state_dict(), './model/' + '_gnet.pth')
                torch.save(dnet.state_dict(), './model/' + '_dnet.pth')
                torch.save(pdnet.state_dict(), './model/' + '_pdnet.pth')
                # print('Model saved.')
            if idx % 200 == 0 or (epoch == opt.epochs -1 and idx == len(custom_loader) - 1):
                print("Epoch: {}, step: {}, loss_valid: {:.5f}, loss_hole: {:.5f}, loss_perceptual: {:.5f}, loss_total: {:.5f}, "
                     "loss_style: {:.5f}, l_gan: {:.5f}, l_d: {:.5f}, l_dp: {:.5f}".format(epoch, idx, np.mean(losses_valid), 
                     np.mean(losses_hole), np.mean(losses_perceptual), np.mean(losses), 
                     np.mean(losses_style), np.mean(losses_gan), np.mean(losses_d), np.mean(losses_dp)))

                losses.clear()
                losses_valid.clear()
                losses_hole.clear()
                losses_perceptual.clear()
                losses_style.clear()
                losses_gan.clear()
                losses_d.clear()
                losses_dp.clear()

                with torch.no_grad():
                    pred = Punet(test_masked, test_mask)
                    pred = pred.detach().cpu()
                    target = test_img.detach().cpu()
                    masked = test_masked.detach().cpu()
                
                    pred = de_normalize(pred)
                    target = de_normalize(target)
                    masked = de_normalize(masked)

                sample_pred = vutils.make_grid(pred, padding=2, normalize=False)
                sample_target = vutils.make_grid(target, padding=2, normalize=False)
                sample_masked = vutils.make_grid(masked, padding=2, normalize=False)
                plt.figure(figsize=(32, 16))
                plt.axis('off')
                plt.title('fake image')
                plt.subplot(3, 1, 1)
                plt.imshow(np.transpose(sample_pred, (1, 2, 0)))
                plt.subplot(3, 1, 2)
                plt.imshow(np.transpose(sample_target, (1, 2, 0)))
                plt.subplot(3, 1, 3)
                plt.imshow(np.transpose(sample_masked, (1, 2, 0)))
                plt.savefig("./sample/epoch_{}_iter_{}.png".format(epoch, idx))
                plt.close()

if __name__ == '__main__':
    print('Initialized.')
    parser = BaseParser()
    opt = parser.parse()
    train(opt)
    print('Over.')
