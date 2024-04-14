import numpy as np
from    matplotlib import pyplot as plt

import os, cv2
from skimage import io, transform
from model.lib.AE_model import AE_Net
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

batch_size = 10
weight = 64
height = 64
input_image_channel = 3
epochs = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def maxmin_norm(data, ):
    data = (data-data.min())/(data.max() - data.min())
    return data

# 旋转数据并限制取值
def pre_processing(features):
    x = (features['X'] - features['Cell X']) / 5
    y = (features['Y'] - features['Cell Y']) / 5

    A = np.radians(features['Azimuth'])
    x_ = x * np.cos(A) - y * np.sin(A)
    y_ = x * np.sin(A) + y * np.cos(A)
    features['x_'] = x_
    features['y_'] = y_
    features = features[features["x_"] > -32]
    features = features[features["x_"] < 32]
    features = features[features["y_"] > 0]
    features = features[features["y_"] < 64]
    return features

def gen_building_map(features):
    og_building_map = -np.ones([64,64])
    RSRP_map = np.zeros([64, 64])
    altitude_map = np.zeros([64, 64])

    x_ = features['x_'].values + 32
    y_ = features['y_'].values
    H_building = features['Building Height'].values
    H_altitude = features['Altitude'].values
    RSRP = features['RSRP'].values
    # value_building_avg = H_building[np.nonzero(H_building)].mean()
    # print(value_building_avg)
    for i in range(len(x_)):
        if x_[i] < 64:
            if y_[i] < 64:
                xx = int(x_[i])
                yy = int(y_[i])
                og_building_map[xx, yy] = H_building[i]
                altitude_map[xx, yy] = H_altitude[i]
                RSRP_map[xx, yy] = RSRP[i]

    RSRP_map[np.isnan(RSRP_map)] = np.nanmean(RSRP_map)
    image_Tx = np.zeros(64, 64)
    image_Tx[0][32] = 1

    return og_building_map,RSRP_map,image_Tx


class SR_dataloader(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 numTx=80,
                 thresh=0.2,
                 carsSimul="no",
                 carsInput="no",
                 ):

        self.sim_dir_dataset = "/data/RadioUnet/"
        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.sim_numTx = numTx
        self.sim_thresh = thresh
        self.sim_dir_gain = self.sim_dir_dataset + "gain/IRT2/"
        self.sim_dir_buildings = self.sim_dir_dataset + "png/buildings_complete/"
        self.sim_dir_Tx = self.sim_dir_dataset + "png/antennas/"


        self.real_dir_dataset = "/data/RSRP_dataset"
        self.file = os.listdir(self.real_dir_dataset + 'train_set/')


    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.sim_numTx

    def __getitem__(self, idx):
        input_dict = {}
        P = random.uniform(0, 2)
        # 取仿真图
        if P >= 0 and P < 1:
            idxr = np.floor(idx / self.sim_numTx).astype(int)
            idxc = idx - idxr * self.sim_numTx
            dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
            # names of files that depend only on the map:
            name1 = str(dataset_map_ind) + ".png"
            # names of files that depend on the map and the Tx:
            name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

            # Load buildings:
            img_name_buildings = os.path.join(self.sim_dir_buildings, name1)
            image_buildings = np.asarray(io.imread(img_name_buildings))

            # Load Tx (transmitter):
            img_name_Tx = os.path.join(self.sim_dir_Tx, name2)
            image_Tx = np.asarray(io.imread(img_name_Tx))

            # Load radio map:
            img_name_gain = os.path.join(self.sim_dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 255

            # pathloss threshold transform
            if self.sim_thresh > 0:
                mask = image_gain < self.sim_thresh
                image_gain[mask] = self.sim_thresh
                image_gain = image_gain - self.sim_thresh * np.ones(np.shape(image_gain))
                image_gain = image_gain / (1 - self.sim_thresh)

            image_buildings = cv2.resize(image_buildings, (64, 64))
            image_Tx = cv2.resize(image_Tx, (64, 64))
            image_gain = cv2.resize(image_gain, (64, 64))

            image_buildings = np.reshape(image_buildings, [64, 64])
            image_Tx = np.reshape(image_Tx, [64, 64])
            image_gain = np.reshape(image_gain, [1, 64, 64])

            mask_rx = image_gain == image_gain.max()
            image_Tx = image_gain * mask_rx
            image_Tx = np.reshape(image_Tx, [64, 64])

            image_buildings = maxmin_norm(image_buildings)
            image_Tx = maxmin_norm(image_Tx)
            image_gain = maxmin_norm(image_gain)

            a = np.zeros(64 * 64, dtype=int)
            a[:500] = 1
            np.random.shuffle(a)
            mask = np.resize(a, [64, 64])
            image_gain_mask = image_gain[0, :, :] * mask
            # make random measured sample

            # inputs to radioUNet
            inputs = np.stack([image_buildings, image_Tx, image_gain_mask], axis=0)
            is_real_old = 0
            input_dict = {'label': inputs, 'inst': is_real_old, 'image': inputs}
        # 取真实图
        if P >= 1 and P < 2:

            features = pd.read_csv("./train_209901.csv")

            features = pre_processing(features)
            # 原始的建筑物高度图 ； 建筑物高度mask矩阵 ； 补全后的建筑物高度图 ；补全后的海拔高度图 ； 原始的RSRP值 ； 原始的海拔高度图
            og_building_map,RSRP_map,image_Tx = gen_building_map(features)
            inputs = np.stack([og_building_map, image_Tx, RSRP_map], axis=0)
            is_real_old = 1

            input_dict = {'label': inputs, 'inst': is_real_old, 'image': inputs}

        return input_dict
