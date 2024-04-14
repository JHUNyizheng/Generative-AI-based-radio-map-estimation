from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import sys
from skimage import io
import cv2
import numpy as np
from torch.utils.data import DataLoader
import time
from src.opts import opts
from src.dataset.dataset_factory import get_dataset
from src.model.model import create_model, load_model, save_model
from src.model.loss import DiceBCELoss

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

""" modelG: rsrp + tx ---> building """
""" modelF: building + tx ---> rsrp """
def test(model,tx_image,rsrp_image,building_image,opt,rate):
    s_mask = sample_mask(1 - building_image, sample_rate=rate / 100)
    # sample_gain = rsrp_image * s_mask - (1 - s_mask)
    sample_gain = rsrp_image
    sample_gain[sample_gain == 1] = -1
    sample_gain[sample_gain == 0] = -1
    # tx = torch.from_numpy(np.expand_dims(np.expand_dims(tx_image.astype(np.float32), axis=0),axis=0)).to(opt.device)
    # rsrp = torch.from_numpy(np.expand_dims(np.expand_dims(rsrp_image.astype(np.float32), axis=0),axis=0)).to(opt.device)
    # building = torch.from_numpy(np.expand_dims(np.expand_dims(building_image.astype(np.float32), axis=0),axis=0)).to(opt.device)
    # sample_gain = torch.from_numpy(np.expand_dims(np.expand_dims(sample_gain.astype(np.float32), axis=0),axis=0)).to(opt.device)

    tx = F.interpolate(torch.Tensor(tx_image).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bicubic',align_corners=True).to(opt.device)
    rsrp = F.interpolate(torch.Tensor(rsrp_image).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bicubic',align_corners=True).to(opt.device)
    building = F.interpolate(torch.Tensor(building_image).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bicubic',align_corners=True).to(opt.device)
    sample_gain = F.interpolate(torch.Tensor(sample_gain).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bicubic',align_corners=True).to(opt.device)
    # building[building != 0] = 1

    if opt.arch == 'Interpolation':
        modelG = model[0].eval()

        r_pred = modelG(torch.cat((sample_gain, tx), 1))
        b_pred = torch.zeros_like(r_pred)
    if opt.arch == 'RadioUnet':
        modelF = model[0].eval()

        r_pred = modelF(torch.cat((building, tx), 1))
        b_pred = torch.zeros_like(r_pred)

    elif opt.arch == 'RadioCycle':

        modelG = model[0].eval()
        modelF = model[1].eval()

        b_pred = modelG(torch.cat((sample_gain, tx), 1))
        # sigmid转换为概率分布
        fake_B = torch.sigmoid(b_pred)
        b_pred = torch.zeros_like(b_pred)
        b_pred[fake_B > 0.5] = 1
        b_pred[fake_B <= 0.5] = 0

        # r_pred = modelF(torch.cat((b_pred, tx), 1))
        r_pred = modelF(torch.cat((building, tx), 1))

    elif 'RadioYnet' in opt.arch:
        modelG = model[0].eval()
        b_pred, r_pred = modelG(torch.cat((sample_gain, tx), 1))
        if 'NRM' in opt.arch:
            r_pred = torch.zeros_like(b_pred)
        b_pred = torch.sigmoid(b_pred)
        b_pred = b_pred > 0.5

    elif opt.arch == 'RadioTrans':
        modelF = model[0].eval()
        r_pred = modelF(torch.cat((building, tx), 1))
        b_pred = torch.zeros_like(r_pred)

    return b_pred[0][0].detach().cpu().numpy(), r_pred[0][0].detach().cpu().numpy() , sample_gain[0][0].detach().cpu().numpy(), rsrp[0][0].detach().cpu().numpy(), building[0][0].detach().cpu().numpy()

if __name__ == '__main__':
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # opt.exp_id = "cycleDWA_s20"
    path = "/data/RadioUnet/"

    if opt.exp_id == 'default':
        print("exp_id null !!!")
        sys.exit(1)
    else:
        opt.model_path = os.path.join('..','..', "results", opt.arch, opt.exp_id)

    """ Load Input Image"""

    # building = '/data/SampleRadioMap/rate50.0p/val/BuildingMap/604.png'
    # tx = '/data/SampleRadioMap/rate50.0p/val/Tx/604_2.png'
    # rsrp = '/data/SampleRadioMap/rate50.0p/val/RadioMap/604_2.png'

    # building = '/data/SampleRadioMap/rate50.0p/train/BuildingMap/199.png'
    # tx = '/data/SampleRadioMap/rate50.0p/train/Tx/199_2.png'
    # rsrp = '/data/SampleRadioMap/rate50.0p/train/RadioMap/199_2.png'

    building = '/data/RadioUnet/RadioMap3DSeer/png/buildingsWHeight/604.png'
    tx = '/data/RadioUnet/RadioMap3DSeer/png/antennasWHeight/604_2.png'
    rsrp = '/data/RadioUnet/RadioMap3DSeer/gain/604_2.png'

    """ Extract the name """
    name = building.split("/")[-1].split(".")[0]

    """ Load Model """
    print('Creating model...')
    model = create_model(opt=opt)
    for i in range(len(model)):
        model[i] = model[i].to(opt.device)

    if opt.load_model != '':
        model = load_model(model, opt.load_model, opt)
    else:
        model = load_model(model, opt.model_path, opt)

    """ Create a directory. """
    opt.image_path = os.path.join('..', '..', "image", opt.arch, opt.exp_id)
    if not os.path.exists(opt.image_path):
        os.makedirs(opt.image_path)

    if opt.dataset == 'real':
        # 21
        id = os.path.join(os.path.join('/data/RSRP_dataset/RadioGAN/dataset/measured_dataset/', 'train'), '25' + '.npy')
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
        building_image = BMsk_map
        tx_image = np.zeros_like(building_image)
        # image_gain = FSRP_map
        image_gain = RSRP_map
        # y = np.nonzero(Pint_map)[1]
        # image_Tx[32][0:y[0]] = 1
        tx_image[32][0] = 1
        tx_image[31][0] = 1
        tx_image[33][0] = 1
        tx_image[32][1] = 1

        building_image = maxmin_norm(building_image)
        tx_image = maxmin_norm(tx_image)
        rsrp_image = maxmin_norm(image_gain)
        building_image[building_image != 0] = 1
        rsrp_image = rsrp_image * (1 - building_image)

    elif opt.dataset == 'simulation' or opt.dataset == 'radiomap3d':
        """ Reading image """
        tx_image = np.asarray(io.imread(tx)) / 255
        rsrp_image = np.asarray(io.imread(rsrp)) / 255
        building_image = np.asarray(io.imread(building)) / 255

    """ testing the model """
    print('Starting testing...')

    per_step = 5
    per1 = 0
    per2 = 100

    for i in range(int((per2 - per1) / per_step)):
        rate = per2 - i * per_step

        start_time = time.time()
        build, rsrp, sample_gain, radio, building= test(model,tx_image,rsrp_image,building_image,opt,rate)
        end_time = time.time()
        data_str = f'Time: {end_time - start_time}s\n'
        print(data_str)
        cat_images = np.concatenate([sample_gain * 255, building * 255, build * 255, rsrp * 255, radio * 255], axis=1)
        # plt.imshow(cat_images)
        # plt.show()
        cv2.imwrite(f"{opt.image_path}/{name}_{rate}.png", cat_images)

    print("Finish!!!")

