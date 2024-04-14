from skimage import io
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision

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

class SimRsrpDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, path, phase):

        self.dir_dataset = path
        self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
        # Determenistic "random" shuffle of the maps:
        np.random.seed(42)
        #np.random.shuffle(self.maps_inds)
        self.numTx = 80
        self.thresh = 0
        self.simulation = "IRT2"
        self.dir_gain = self.dir_dataset + "gain/IRT2/"  # RSRP的地址
        self.dir_buildings = self.dir_dataset + "png/buildings_complete/" # building的地址
        self.dir_Tx = self.dir_dataset + "png/antennas/" # 基站位置的地址
        self.sample_rate = 0.05

        # self.A_paths = os.listdir(self.dir_gain)
        # self.B_paths = os.listdir(self.dir_buildings)

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            # self.ind2 = 699
            self.ind2 = 699

        self.phase = phase

        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

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
        idxr = np.floor(index / self.numTx).astype(int)
        idxc = index - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"


        # Load buildings:
        img_name_buildings = os.path.join(self.dir_buildings, name1)
        image_buildings = np.asarray(io.imread(img_name_buildings))

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))

        # Load radio map:
        img_name_gain = os.path.join(self.dir_gain, name2)
        image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 255

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        #image_buildings = cv2.resize(image_buildings, (256, 256))
        ## image_Tx = cv2.resize(image_Tx, (64, 64))
        #image_gain = cv2.resize(image_gain, (256, 256))
        #
        image_buildings = np.reshape(image_buildings, [256, 256])
        # image_Tx = np.reshape(image_Tx, [64, 64])
        image_gain = np.reshape(image_gain, [256, 256])
        #
        mask_rx = image_gain == image_gain.max()
        image_Tx = image_gain * mask_rx
        image_Tx = np.reshape(image_Tx, [256, 256])
        #
        image_buildings = maxmin_norm(image_buildings)
        image_Tx = maxmin_norm(image_Tx)
        image_gain = maxmin_norm(image_gain)
        image_buildings[image_buildings != 0] = 1

        if self.phase == 'train':
            sample_rate = np.random.randint(5,20)/100
            s_mask = sample_mask(1 - image_buildings, sample_rate=sample_rate)
        else:
            s_mask = sample_mask(1 - image_buildings, sample_rate=self.sample_rate)

        sample_gain = image_gain * s_mask - (1-s_mask)
        #sample_gain = image_gain
        #ml_image = ML_interpolation(image_Tx, 1 - s_mask * (1 - image_buildings), sample_gain, image_gain, mode='KNN')

        # input_img = np.stack((sample_gain,image_Tx),axis=2)
        # # input_img = np.stack((image_buildings, image_Tx), axis=2)
        # input_img = np.transpose(input_img,[2,0,1])

        self.sample_rate = 0.05
        s_mask = sample_mask(1 - image_buildings, sample_rate=self.sample_rate)
        sample_gain = image_gain * s_mask - (1 - s_mask)
        cv2.imwrite('building.png', image_buildings * 255)
        cv2.imwrite('1.png', sample_gain * 255)


        self.sample_rate = 0.10
        s_mask = sample_mask(1 - image_buildings, sample_rate=self.sample_rate)
        sample_gain = image_gain * s_mask - (1 - s_mask)
        box = np.array([0, 0, 50, 50])
        sample_gain[50:,50:] = 0
        cv2.imwrite('2.png', sample_gain * 255)

        self.sample_rate = 0.15
        s_mask = sample_mask(1 - image_buildings, sample_rate=self.sample_rate)
        sample_gain = image_gain * s_mask - (1 - s_mask)
        box = np.array([60, 60, 100, 100])
        sample_gain[0:60, :] = 0
        sample_gain[100:, :] = 0
        sample_gain[:, 0:60] = 0
        sample_gain[:, 100:] = 0
        cv2.imwrite('3.png', sample_gain * 255)

        self.sample_rate = 0.2
        s_mask = sample_mask(1 - image_buildings, sample_rate=self.sample_rate)
        sample_gain = image_gain * s_mask - (1 - s_mask)
        box = np.array([200, 20, 250, 150])
        sample_gain[0:200, :] = 0
        sample_gain[250:, :] = 0
        sample_gain[:, 0:20] = 0
        sample_gain[:, 150:] = 0
        cv2.imwrite('4.png', sample_gain * 255)


        image_buildings = np.reshape(image_buildings, [1, 256, 256])
        image_gain = np.reshape(image_gain,[1, 256,256])
        image_Tx = np.reshape(image_Tx, [1, 256, 256])
        sample_gain = np.reshape(sample_gain, [1, 256, 256])







        return image_buildings.astype(np.float), \
               image_gain.astype(np.float),\
               sample_gain.astype(np.float), \
               image_Tx.astype(np.float)

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def SampleRate(self,rate):
        self.sample_rate = rate / 100
