"""
 @file: train.py
 @Time    : 2023/1/11
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
from torch.nn import MSELoss
from torch.optim import Adadelta
from models import EncoderDecoder
import matplotlib.pyplot as plt
import cv2

DATA_ROOT = "./data"
SAVE_ROOT = "./saved"
MEAN = (0.1307,)
STD = (0.3081,)
DATASET_RATIO = 0.2
BATCHSIZE = 64
FIXED_LENGTH = 4
EPOCHS = 10
LR = 1
SAVE_FREQ = 100


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

    train_set = MyDataset(trainset, ratio=DATASET_RATIO)
    val_set = MyDataset(valset, ratio=DATASET_RATIO)

    train_loader = DataLoader(train_set
                              , batch_size=BATCHSIZE
                              , shuffle=True
                              , num_workers=4)

    val_loader = DataLoader(val_set
                            , batch_size=BATCHSIZE
                            , shuffle=True
                            , num_workers=4)

    model = EncoderDecoder()
    criterion = MSELoss()
    optim = Adadelta(model.parameters(), LR)

    # training and validation
    for epoch in range(EPOCHS):
        for i, (data, label) in tqdm.tqdm(enumerate(train_loader)):
            model.train()
            model = model.to(device)
            data = data.to(device)
            label = label.to(device)

            output = model(data)

            loss = criterion(output, label)
            print(f"Train-Epoch: {epoch}, step:{i}, loss:{loss.item()}")

            optim.zero_grad()
            loss.backward()
            optim.step()

            # every 100 steps, check to what extent the denoise ability is trained
            if i % SAVE_FREQ == 0:
                with torch.no_grad():
                    batch_img_show_and_save(data, "noise", "train", epoch, i)
                    batch_img_show_and_save(output, "denoise", "train", epoch, i)

        # evaluation
        for i, (data, label) in tqdm.tqdm(enumerate(val_loader)):
            model.eval()
            model = model.to(device)
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            print(f"Val-Epoch: {epoch}, step: {i}, loss: {loss.item()}")

            if i % SAVE_FREQ == 0:
                with torch.no_grad():
                    batch_img_show_and_save(data, "noise", "val", epoch, i)
                    batch_img_show_and_save(output, "denoise", "val", epoch, i)


if __name__ == '__main__':
    main()
