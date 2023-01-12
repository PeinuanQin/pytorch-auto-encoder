"""
 @file: vae_train.py
 @Time    : 2023/1/12
 @Author  : Peinuan qin
 """

import os.path
import torch.cuda
import torchvision
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from util import MyDataset
from torch.nn import MSELoss, KLDivLoss
from torch.distributions import kl_divergence, Normal
from torch.optim import Adadelta, Adam
from models import VAE
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

DATA_ROOT = "./data"
SAVE_ROOT = "./saved_vae"
MEAN = (0.1307,)
STD = (0.3081,)
DATASET_RATIO = 1
BATCHSIZE = 64
FIXED_LENGTH = 4
EPOCHS = 10
LR = 0.001
SAVE_FREQ = 100
PRINT_FREQ = 100


def transfer_to_uint8(img):
    """
    transfer img whose value domain to [0-255] so that we can use
    opencv to save the image
    :param img:
    :return:
    """
    if isinstance(img, np.ndarray):
        img -= np.min(img)
        img /= np.max(img)
        img *= 255
        return img


def batch_img_show_and_save(batch_tensor
                            , prefix
                            , stage
                            , epoch
                            , step
                            , show=False):
    """

    :param batch_tensor: a batch of images predicted by the decoder
    :param prefix: the prefix to name the saving files
    :param stage: train / val
    :param epoch: current epoch
    :param step: current step
    :param show: whether to display the generated results
    :return:
    """
    np_output = np.copy(batch_tensor)
    imgs = torchvision.utils.make_grid(torch.from_numpy(np_output), nrow=8, padding=1)
    # print(imgs.shape)
    # (c, h, w) -> (h, w, c)
    imgs = np.moveaxis(imgs.numpy(), 0, 2)
    if show:
        plt.axis('off')
        plt.imshow(imgs)
        plt.show()

    # adjust r, g, b -> b, g, r, so that the opencv can save the image
    cv_imgs = imgs[:, :, ::-1]

    save_dir = os.path.join(SAVE_ROOT, stage)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{prefix}_{stage}_{str(epoch).zfill(FIXED_LENGTH)}_{str(step).zfill(FIXED_LENGTH)}.jpg"
    save_file_path = os.path.join(save_dir, filename)

    # transfer image to 0-255 and use cv2 to imwrite
    cv_imgs = np.uint8(transfer_to_uint8(cv_imgs))
    cv2.imwrite(save_file_path, cv_imgs)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare raw dataset, transfer raw data to tensor, so we can use numpy to tackle it
    trainset = MNIST(DATA_ROOT
                     , train=True
                     , transform=transforms.Compose([transforms.ToTensor()
                                                        , transforms.Normalize(MEAN, STD)])
                     , download=True)

    valset = MNIST(DATA_ROOT
                   , train=False
                   , transform=transforms.Compose([transforms.ToTensor()
                                                      , transforms.Normalize(MEAN, STD)])
                   , download=False)

    # only use 0.2 of the raw data for training and validation

    train_set = MyDataset(trainset, ratio=DATASET_RATIO, add_noise=False)
    val_set = MyDataset(valset, ratio=DATASET_RATIO, add_noise=False)

    train_loader = DataLoader(train_set
                              , batch_size=BATCHSIZE
                              , shuffle=True
                              , num_workers=1)

    val_loader = DataLoader(val_set
                            , batch_size=BATCHSIZE
                            , shuffle=True
                            , num_workers=1)

    model = VAE()
    # mse_criterion = MSELoss()
    # mse_criterion = F.binary_cross_entropy
    mse_criterion = F.binary_cross_entropy_with_logits
    kl_criterion = kl_divergence
    optim = Adam(model.parameters(), LR)

    # training and validation
    for epoch in range(EPOCHS):
        for i, (data, label) in tqdm.tqdm(enumerate(train_loader)):
            model.train()
            model = model.to(device)
            data = data.to(device)
            label = label.to(device)

            distribution, output = model(data)

            # use normal distribution N~(0,1) to regularize the output distribution of the encoder
            distribution_loss = kl_criterion(distribution, Normal(0, 1)).sum(-1).mean()

            # re-constructive loss of the decoder generation
            # reconstructive_loss = mse_criterion(output, label)
            reconstructive_loss = mse_criterion(output, label, size_average=False)

            alpha = 1
            beta = 1

            vae_loss = alpha * distribution_loss + beta * reconstructive_loss

            if i % PRINT_FREQ == 0:

                print(f"Train-Epoch: {epoch}"
                      f", step:{i}"
                      f", vae_loss:{vae_loss.item()}"
                      f", distribution_loss: {distribution_loss.item()}"
                      f", reconstructive_loss: {reconstructive_loss.item()}")

            optim.zero_grad()
            vae_loss.backward()
            optim.step()

            # every 100 steps, check to what extent the denoise ability is trained
            if i % SAVE_FREQ == 0:
                with torch.no_grad():
                    batch_img_show_and_save(data.cpu(), "raw", "train", epoch, i)
                    batch_img_show_and_save(output.cpu(), "gen", "train", epoch, i)

        # evaluation
        for i, (data, label) in tqdm.tqdm(enumerate(val_loader)):
            model.eval()
            model = model.to(device)
            data = data.to(device)
            label = label.to(device)

            distribution, output = model(data)

            # use normal distribution N~(0,1) to regularize the output distribution of the encoder
            distribution_loss = kl_criterion(distribution, Normal(0, 1)).sum(-1).mean()

            # re-constructive loss of the decoder generation
            # reconstructive_loss = mse_criterion(output, label)
            reconstructive_loss =mse_criterion(output, label, size_average=False)

            alpha = 1
            beta = 1

            vae_loss = alpha * distribution_loss + beta * reconstructive_loss

            if i % PRINT_FREQ == 0:
                print(f"Val-Epoch: {epoch}"
                      f", step:{i}"
                      f", vae_loss:{vae_loss.item()}"
                      f", distribution_loss: {distribution_loss.item()}"
                      f", reconstructive_loss: {reconstructive_loss.item()}")

            if i % SAVE_FREQ == 0:
                with torch.no_grad():
                    batch_img_show_and_save(data.cpu(), "raw", "val", epoch, i)
                    batch_img_show_and_save(output.cpu(), "gen", "val", epoch, i)


if __name__ == '__main__':
    main()
