from skimage import io
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision
from skimage import io
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

def maxmin_norm(data, ):
    data = (data-data.min())/(data.max() - data.min())
    return data

def sample_mask(mask, sample_rate):
    id = np.transpose(np.nonzero(mask))
    id_idx = np.arange(len(id))
    random_id = np.random.choice(id_idx, int(len(id_idx) * sample_rate))

    b = np.zeros_like(mask)
    for sample_id in id[random_id]:
        b[sample_id[0],sample_id[1]] = 1
    return b


class RealRsrpDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, path, phase):

        self.trainset_path = '/data/RSRP_dataset/RadioGAN/dataset/measured_dataset/'
        self.testset_path = '/data/RSRP_dataset/RadioGAN/dataset/measured_dataset/'

        self.dir_dataset = path
        self.maps_inds = np.arange(0, 170, 1, dtype=np.int16)
        self.sample_rate = 0.05
        # Determenistic "random" shuffle of the maps:
        # np.random.seed(42)
        #np.random.shuffle(self.maps_inds)

        self.numTx = 1

        self.thresh = 0
        self.simulation = "IRT2"
        self.dir_gain = self.dir_dataset + "gain/IRT2/"  # RSRP的地址
        self.dir_buildings = self.dir_dataset + "png/buildings_complete/" # building的地址
        self.dir_Tx = self.dir_dataset + "png/antennas/" # 基站位置的地址

        # self.A_paths = os.listdir(self.dir_gain)
        # self.B_paths = os.listdir(self.dir_buildings)

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 130
        elif phase == "val":
            self.ind1 = 131
            self.ind2 = 149
        elif phase == "test":
            self.ind1 = 0
            self.ind2 = 29

        self.phase = phase

        if(self.phase == 'train' or self.phase == 'val'):
            self.dir = os.path.join(self.trainset_path, 'train')  # create a path '/path/to/data/trainA'
        else:
            self.dir = os.path.join(self.trainset_path, 'test')  # create a path '/path/to/data/trainA'

        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        id = os.path.join(self.dir,str(index) + '.npy')
        data = np.load(id)

        height_map = data[0, :, :]
        mask = data[1, :, :]
        resi_map = data[2, :, :]
        BMsk_map = data[3, :, :]
        RSRP_map = data[4, :, :]
        SPMp_map = data[5, :, :]
        FSRP_map = data[6, :, :]
        Pint_map = data[7, :, :]
        Alti_map = data[8, :, :]

        # print(data)
        image_buildings = BMsk_map
        image_Tx = np.zeros_like(image_buildings)
        image_gain = FSRP_map
        # y = np.nonzero(Pint_map)[1]
        # image_Tx[32][0:y[0]] = 1
        image_Tx[32][0] = 1
        image_Tx[31][0] = 1
        image_Tx[33][0] = 1
        image_Tx[32][1] = 1

        image_buildings = maxmin_norm(image_buildings)
        image_Tx = maxmin_norm(image_Tx)
        image_gain = maxmin_norm(image_gain)
        image_buildings[image_buildings != 0] = 1
        image_gain = image_gain * (1 - image_buildings)

        # image_gain = 2 * image_gain - 1
        # image_Tx = (image_Tx * 255).transpose(1,2,0)
        # image_sample_gain = (image_sample_gain * 255).transpose(1,2,0)
        # image_buildings = (image_buildings * 255).transpose(1,2,0)
        # image_gain = (image_gain * 255).transpose(1,2,0)

        # image = np.concatenate([image_Tx,image_buildings,image_gain],axis=2)
        #
        # # rsrp = image_sample_gain
        # # build =image_buildings
        # # radio = image_gain
        # # tx = image_Tx
        # # rsrp = np.expand_dims(rsrp[0], axis=2)
        # # build = np.expand_dims(build[0], axis=2)
        # # radio = np.expand_dims(radio[0], axis=2)
        # # tx = np.expand_dims(tx[0], axis=2)
        # #
        # # cat_images = np.concatenate(
        # #     [rsrp, build, radio,tx], axis=1
        # # )
        # # plt.axis('off')
        # # plt.imshow(cat_images, cmap='viridis')
        # # plt.show()
        # im_aug = transforms.Compose([
        #     # transforms.RandomCrop(32, padding=4),  # 随机裁剪
        #     transforms.RandomRotation((0, 45)),
        #     transforms.RandomHorizontalFlip(p=1),
        #     transforms.RandomVerticalFlip(p=1)
        #     # transforms.ToTensor(),
        # ])
        # image = transforms.ToPILImage()(image.astype(np.uint8))
        # image = im_aug(image)
        # image = np.array(image)
        #
        #
        # image_Tx = image[:, :, 0]
        # # image_sample_gain = image[:, :, 1]
        # image_buildings = image[:, :, 1]
        # image_gain = image[:, :, 2]
        #
        # image_buildings = maxmin_norm(image_buildings)
        # image_Tx = maxmin_norm(image_Tx)
        # # image_buildings = maxmin_norm(image_buildings)
        # image_gain = maxmin_norm(image_gain)


        if self.phase == 'train':
            sample_rate = np.random.randint(5, 50)/100
            # sample_rate = 0.1
            s_mask = sample_mask(1 - image_buildings , sample_rate=sample_rate)
        else:
            s_mask = sample_mask(1 - image_buildings, sample_rate=self.sample_rate)

        sample_gain = image_gain * s_mask - (1-s_mask)


        image_Tx = np.array(F.interpolate(torch.Tensor(image_Tx).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=True).squeeze(0))
        sample_gain = np.array(F.interpolate(torch.Tensor(sample_gain).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=True).squeeze(0))
        image_buildings = np.array(F.interpolate(torch.Tensor(image_buildings).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=True).squeeze(0))
        image_gain = np.array(F.interpolate(torch.Tensor(image_gain).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=True).squeeze(0))
        image_buildings[image_buildings != 0] = 1

        # plt.imshow(image_Tx[0])
        # plt.show()
        #
        # plt.imshow(sample_gain[0])
        # plt.show()
        #
        # plt.imshow(image_buildings[0])
        # plt.show()
        #
        # plt.imshow(image_gain[0])
        # plt.show()


        # image_Tx = np.reshape(image_Tx, [1, 256, 256])
        # sample_gain = np.reshape(sample_gain, [1, 256, 256])
        # image_buildings = np.reshape(image_buildings, [1, 256, 256])
        # image_gain = np.reshape(image_gain, [1, 256, 256])

        # plt.imshow(image_gain)
        # plt.show()
        #
        # plt.imshow(sample_gain)
        # plt.show()
        # print('OK')

        return image_buildings, image_gain, sample_gain, image_Tx


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def SampleRate(self, rate):
        self.sample_rate = rate / 100