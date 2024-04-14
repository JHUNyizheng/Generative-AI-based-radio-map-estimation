import numpy as np
from    matplotlib import pyplot as plt

import os, cv2
from skimage import io, transform
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def maxmin_norm(data, ):
    data = (data-data.min())/(data.max() - data.min())
    return data

def maxmin_norm_mask(data, mask):
    data = (data-data[mask.astype('bool')].min()) \
                 / (data[mask.astype('bool')].max() - data[mask.astype('bool')].min())

    # data = (data-0.5)*2
    data = data * mask

    return data

def knn_missing_filled(x_train, y_train, test, k=3, dispersed=True):
    if dispersed:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    else:
        clf = KNeighborsRegressor(n_neighbors=k, weights="distance")

    clf.fit(x_train, y_train)
    return clf.predict(test)

def missing_filled(x_train, y_train, test, k=3, dispersed=True):

    predict = knn_missing_filled(x_train, y_train, test, k=k, dispersed=dispersed)
    return predict

def img_imputer(input_image, k):
    building_mask_map = np.copy(input_image)
    building_height_map = np.copy(input_image)
    building_mask_map[input_image > 0] = 0
    building_mask_map[input_image > 5] = 1
    trainx_cls = []
    trainy_cls = []
    testx_cls = []
    trainx_reg = []
    trainy_reg = []
    testx_reg = []
    for i in range(64):
        for j in range(64):
            if building_mask_map[i, j] == -1:
                testx_cls.append([i, j])
            elif building_mask_map[i, j] == 0:
                trainx_cls.append([i, j])
                trainy_cls.append(0)
            elif building_mask_map[i, j] == 1:
                trainx_cls.append([i, j])
                trainy_cls.append(1)

    testy_cls = missing_filled(trainx_cls, trainy_cls, testx_cls, k=k, dispersed=True)
    # testy_cls = knn_missing_filled(trainx_cls, trainy_cls, testx_cls, k=k, dispersed=True)

    for n in range(len(testx_cls)):
        building_mask_map[testx_cls[n][0], testx_cls[n][1]] = testy_cls[n]

    for i in range(64):
        for j in range(64):
            if building_mask_map[i, j] == 1:
                if building_height_map[i, j] == -1:
                    testx_reg.append([i, j])
                else:
                    trainx_reg.append([i, j])
                    trainy_reg.append(building_height_map[i, j])

    testy_reg = missing_filled(trainx_reg, trainy_reg, testx_reg, k=1, dispersed=False)
    # testy_reg = knn_missing_filled(trainx_reg, trainy_reg, testx_reg, k=1, dispersed=False,type='knn')
    for n in range(len(testx_reg)):
        building_height_map[testx_reg[n][0], testx_reg[n][1]] = testy_reg[n]

    building_height_map[building_height_map == -1] = 0

    out_mask = building_mask_map
    out_img = building_height_map

    return out_mask, out_img

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
    image_Tx = np.zeros([64, 64])
    image_Tx[32][0] = 1

    building_mask_map, building_height_map = img_imputer(og_building_map, k=3)

    return building_mask_map,RSRP_map,image_Tx

# 用于测试
from torch.utils.data import Dataset, DataLoader
class real_dataloader(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, phase="test"):

        self.phase=phase

        self.real_dir_dataset = "/data/RSRP_dataset/"
        self.file = os.listdir(self.real_dir_dataset + 'train_set/')



    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        file_name = self.file[idx]
        features = pd.read_csv(self.real_dir_dataset + 'train_set/' + file_name)
        features = pre_processing(features)
        # 原始的建筑物高度图 ； 建筑物高度mask矩阵 ； 补全后的建筑物高度图 ；补全后的海拔高度图 ； 原始的RSRP值 ； 原始的海拔高度图
        building_mask_map, RSRP_map, image_Tx = gen_building_map(features)
        RSRP_mask = np.copy(RSRP_map)
        RSRP_mask[RSRP_mask != 0] = 1

        RSRP_map[RSRP_map < -100] = -100
        RSRP_map[RSRP_map > -75] = -75
        RSRP_map = maxmin_norm_mask(RSRP_map,RSRP_mask)

        image_building = np.copy(building_mask_map)
        image_building[image_building == 0] = 2
        image_building[image_building < 2] = 0
        image_building[image_building == 2] = 1

        RSRP_map = RSRP_map * image_building

        inputs = np.stack([building_mask_map, image_Tx, RSRP_map], axis=0)
        is_real_old = 1

        RSRP_map = np.reshape(RSRP_map, [1, 64, 64])
        input_dict = {'label': inputs, 'inst': is_real_old, 'image': inputs}

        # inputs to radioUNet

        # inputs = np.stack([og_building_map , image_Tx, RSRP_map], axis=0)

        return [inputs, RSRP_map]

def RMSE(A,B):
    return np.sqrt(np.mean(np.power((A - B), 2)))

