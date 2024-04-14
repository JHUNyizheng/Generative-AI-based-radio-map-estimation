import os, time, pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore")

def get_filename(root_dir, debug=False):
    filenames = []
    sample_cnt = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            sample_cnt += 1
            # print("files: ",os.path.join(root, name),name)
            file_name = name
            file_content = os.path.join(root, name)
            filenames.append([file_name,file_content])
            # if debug:
            #     if sample_cnt == 1:
            #         break
    return filenames

def pre_processing(pb_data):
    x = pb_data['X'] - pb_data['Cell X']
    y = pb_data['Y'] - pb_data['Cell Y']

    x = x / 5
    y = y / 5
    A = np.radians(pb_data['Azimuth'])
    x_ = x * np.cos(A) - y * np.sin(A)
    y_ = x * np.sin(A) + y * np.cos(A)
    pb_data['x_'] = x_
    pb_data['y_'] = y_
    pb_data = pb_data.iloc[list(pb_data["x_"] > -32)]
    pb_data = pb_data.iloc[list(pb_data["x_"] < 32)]
    pb_data = pb_data.iloc[list(pb_data["y_"] > 0)]
    pb_data = pb_data.iloc[list(pb_data["y_"] < 64)]
    return pb_data


from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def knn_missing_filled(x_train, y_train, test, k=3, dispersed=True):
    if dispersed:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    else:
        clf = KNeighborsRegressor(n_neighbors=k, weights="distance")

    clf.fit(x_train, y_train)
    return clf.predict(test)

from sklearn.impute import KNNImputer
def img_imputer(input_image, k, Imputer_of='building' ):
    if Imputer_of == 'building':
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
                if building_mask_map[i,j] == -1:
                    testx_cls.append([i,j])
                elif building_mask_map[i,j] == 0:
                    trainx_cls.append([i,j])
                    trainy_cls.append(0)
                elif building_mask_map[i,j] == 1:
                    trainx_cls.append([i,j])
                    trainy_cls.append(1)
        testy_cls = knn_missing_filled(trainx_cls, trainy_cls, testx_cls, k=k, dispersed=True)
        for n in range(len(testx_cls)):
            building_mask_map[testx_cls[n][0],testx_cls[n][1]]=testy_cls[n]

        for i in range(64):
            for j in range(64):
                if building_mask_map[i,j] == 1:
                    if building_height_map[i, j] == -1:
                        testx_reg.append([i,j])
                    else:
                        trainx_reg.append([i,j])
                        trainy_reg.append(building_height_map[i,j])


        testy_reg = knn_missing_filled(trainx_reg, trainy_reg, testx_reg, k=1, dispersed=False)
        for n in range(len(testx_reg)):
            building_height_map[testx_reg[n][0], testx_reg[n][1]] = testy_reg[n]

        building_height_map[building_height_map==-1] = 0

        out_mask = building_mask_map
        out_img = building_height_map

    elif Imputer_of == 'altitude':
        mask_map = np.copy(input_image)
        altitude_map = np.copy(input_image)
        trainx_reg = []
        trainy_reg = []
        testx_reg = []
        for i in range(64):
            for j in range(64):
                if altitude_map[i, j] == 0:
                    testx_reg.append([i, j])
                else:
                    trainx_reg.append([i, j])
                    trainy_reg.append(altitude_map[i,j])

        testy_reg = knn_missing_filled(trainx_reg, trainy_reg, testx_reg, k=1, dispersed=False)
        for n in range(len(testx_reg)):
            altitude_map[testx_reg[n][0], testx_reg[n][1]] = testy_reg[n]

        mask_map[mask_map != 0] = 1
        out_mask = mask_map
        out_img = altitude_map

    elif Imputer_of == 'RSRP':
        mask_map = np.copy(input_image)
        rsrp_map = np.copy(input_image)
        trainx_reg = []
        trainy_reg = []
        testx_reg = []
        for i in range(64):
            for j in range(64):
                if rsrp_map[i, j] == 0:
                    testx_reg.append([i, j])
                else:
                    trainx_reg.append([i, j])
                    trainy_reg.append(rsrp_map[i, j])

        testy_reg = knn_missing_filled(trainx_reg, trainy_reg, testx_reg, k=k, dispersed=False)
        for n in range(len(testx_reg)):
            rsrp_map[testx_reg[n][0], testx_reg[n][1]] = testy_reg[n]

        mask_map[mask_map != 0] = 1
        out_mask = mask_map
        out_img = rsrp_map

    else:
        print("img_imputer ERROR! Imputer_of what? now is ",Imputer_of)
        out_mask = None
        out_img = None

    return out_mask, out_img



def SPM(features,RSRP_map, altitude_map):
    att_min = features['Altitude'].min()
    f = features['Frequency Band'].values[0]
    h_bs = features['Cell Altitude'].values[0] + \
           features['Cell Building Height'].values[0] + \
           features['Height'].values[0] - att_min
    RSP = features['RS Power'].values[0]

    diff = []

    K1, K2, K3, K4, K5, K6 = 23.5, 44.9, 5.83, 1, -6.55, 0

    K_clutter = 1

    Diffraction = 0.2

    spm_map = np.zeros([64,64])
    #BS coordinate
    x = 32*5
    y = 0
    for i in range(64):
        for j in range(64):
            x_ = i*5
            y_ = j*5
            d = np.sqrt(np.square(x_ - x) + np.square(y_ - y))

            #Rx effect height, assue 1m s.j. flat
            # h_ue = 1
            h_ue = altitude_map[i,j] - altitude_map.min()
            PL = K1 + K2 * np.log10(d) + K3 * np.log10(h_bs) + K4 * Diffraction + \
                 K5 * np.log10(h_bs) * np.log10(d) + K6 * h_ue + K_clutter
            rsrp = RSP - PL
            spm_map[i,j]=rsrp

            if RSRP_map[i, j] !=0:
                # print(len(diff))
                diff.append(rsrp - RSRP_map[i, j])

        # PL = 150

    pl = []
    for i in range(64):
        for j in range(64):
            x_ = i*5
            y_ = j*5
            d = np.sqrt(np.square(x_ - x) + np.square(y_ - y))
            if d > 10:
                pl.append(spm_map[i, j])


    for i in range(64):
        for j in range(64):
            x_ = i*5
            y_ = j*5
            d = np.sqrt(np.square(x_ - x) + np.square(y_ - y))
            if d <= 10:
                spm_map[i, j] = np.max(pl)

    spm_map[np.isnan(spm_map)]=np.nanmean(pl)
    # print("pl：", pl[0])
    # print(np.nanmedian(np.array(diff)))
    spm_map = spm_map - np.nanmedian(np.array(diff))
    # print("pred_RSRP:", np.mean(pred_RSRP), "RSRP:", np.mean(RSRP), np.shape(pred_RSRP))
    # map_rsrp_pred, map_building_mask = get_map(features, pred_RSRP, w, h)
    #
    # map_rsrp_pred = map_rsrp_pred - (map_building_mask * 3.365)

    # if cnt == 4:
    #     generate_images(map_rsrp_pred, map_rsrp, map_deltaH, 'SPM')

    return spm_map


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

                rs = RSRP[i]
                if RSRP[i] > -75:
                    rs = -75
                elif RSRP[i] < -100:
                    rs = -100
                RSRP_map[xx, yy] = rs

                # map_rsrp_pred[xx, yy] = 1
                # if H_building[i] != 0:
                #     map_building_mask[xx, yy] = 1

    # building_map = building_map[building_map > -1]
    # building_map_ = -np.ones([64, 64])

    if np.sum(og_building_map) > 0:

        building_mask_map, building_height_map = img_imputer(og_building_map,k=3,Imputer_of='building')

    else:
        building_mask_map = np.zeros([64,64])
        building_height_map = np.zeros([64, 64])

    _, altitude_height_map = img_imputer(altitude_map, k=1, Imputer_of='altitude')

    building_mask_map[np.isnan(building_mask_map)] = 0
    building_height_map[np.isnan(building_height_map)] = 0
    altitude_height_map[np.isnan(altitude_height_map)] = np.nanmean(altitude_height_map)
    RSRP_map[np.isnan(RSRP_map)] = np.nanmean(RSRP_map)


    return og_building_map,building_mask_map,building_height_map,altitude_height_map,RSRP_map


def cross_point(features):
    theta = features['Electrical Downtilt'].values[0] + features['Mechanical Downtilt'].values[0]
    att_max = features['Altitude'].max()
    h_bs = features['Cell Altitude'].values[0] + \
           features['Cell Building Height'].values[0] + \
           features['Height'].values[0] - att_max
    if theta == 0:
        mapL = 63
    else:
        mapL = int((h_bs/np.tan(theta*np.pi/180))/5)


    if mapL>=64:
        mapL = 63

    cross_point_map = np.zeros([64,64])
    cross_point_map[32,mapL] = 1

    return cross_point_map



# filenames = get_filename("../train_set/")
filenames = get_filename("/data/RSRP_dataset/train_set")


import cv2
cnt=0
# img_save_path = '/data/RSRP_dataset/RadioGAN/dataset/imgs/test/'
for name, context in filenames:

    features = pd.read_csv(context)
    # print(name,cnt,"len:", len(features))
    features=pre_processing(features)
    print(name,cnt,"len:", len(features),len(features)/(64*64))
    # gen_building_map(features)
    # maps = gen_building_map(features)
    og_building_map,building_mask_map,building_height_map,altitude_height_map,RSRP_map = gen_building_map(features)
    SPM_map = SPM(features,RSRP_map, altitude_map=altitude_height_map)
    # maps.append(SPM_map)
    maps = [og_building_map,building_mask_map,building_height_map,altitude_height_map,
            RSRP_map,SPM_map]
    map_labels = ['Original Building Height',
                  'Building Mask Map',
                  'Building Height Map',
                  'Altitude Map',
                  'RSRP Map',
                  'SPM Prediction Map',
                  ]
    RSRP_mask_map = RSRP_map!=0
    residual_RSRP_map = np.zeros([64,64])
    residual_RSRP_map[RSRP_mask_map] = RSRP_map[RSRP_mask_map]-SPM_map[RSRP_mask_map]

    maps.append(residual_RSRP_map)
    map_labels.append('Residual RSRP Map')

    # _, fill_RSRP_map = img_imputer(RSRP_map, k=5, Imputer_of='RSRP')

    # fill_residual_RSRP_map = np.zeros([64, 64])
    # fill_residual_RSRP_map = fill_RSRP_map - SPM_map

    cross_point_map = cross_point(features)




    # N=len(map_labels)
    # fig = plt.figure(figsize=(64, 64))
    # fig, axs = plt.subplots(nrows=1, ncols=N, figsize=(N*3, 3))
    # for i in range(N):
    #     ax = axs[i]
    #     # ax.axis("off")
    #     ax.set_title(map_labels[i])
    #     # plt.colorbar()
    #     im = ax.imshow(maps[i])
    #
    # # plt.tight_layout()
    # plt.show()

    # if np.sum(building_mask_map)==0:
    if len(features)>2000:

        plt.imshow(cross_point_map)
        plt.colorbar()
        plt.show()

        # a = np.abs(building_mask_map-1)
        a = RSRP_mask_map
        a = np.where(a == 0, np.nan, a)
        plt.imshow(RSRP_map*a)
        plt.colorbar()
        plt.show()

        cnt += 1


    # np.save(img_save_path + name[:-4] + '_BMsk' + '.npy', building_mask_map)
    # np.save(img_save_path +  name[:-4] + '_BHgt' + '.npy', building_height_map)
    # np.save(img_save_path +  name[:-4] + '_Alti' + '.npy', altitude_height_map)
    # np.save(img_save_path +  name[:-4] + '_RSRP' + '.npy', RSRP_map)
    # np.save(img_save_path +  name[:-4] + '_SPMp' + '.npy', SPM_map)
    # np.save(img_save_path + name[:-4] + '_resi' + '.npy', residual_RSRP_map)
    # np.save(img_save_path + name[:-4] + '_Fsrp' + '.npy', fill_RSRP_map)
    # np.save(img_save_path + name[:-4] + '_Fres' + '.npy', fill_residual_RSRP_map)
    # np.save(img_save_path + name[:-4] + '_Pint' + '.npy', cross_point_map)

    # if cnt > 5:
    #     # cnt+=1
    #     break
