from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import os
# Ignore warnings
import warnings

#from lib import RadioUNet_modules3, RadioUNet_loaders2
import lib.AE_model as AE_model
import lib.transferNet as transferNet

from lib.dataloader.real_dataloader import real_dataloader
from lib.dataloader.sim_dataloader import sim_dataloader
from lib.dataloader.part2all_brt_dataloader import part2all_brt_dataloader

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

## Load Model and Summary
from torchsummary import summary

def load_model(model, model_path, opt, optimizer=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}

if __name__ == '__main__':
  # load_name = "./result/Trained_Model_tmp.pt"
  # load_name = "./result/Trained_Model_tmp_part2all.pt"

  load_name = "./result/Trained_Model_tmp_building.pt"

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  # Radio_test = transferNet.RadioUNet_c(phase="test")
  # Radio_test = real_dataloader(phase="test")
  Radio_test = sim_dataloader(phase="test")
  # Radio_test = part2all_brt_dataloader(phase="train")

  dataloaders = {
    'test': DataLoader(Radio_test, batch_size=1, shuffle=True, num_workers=0)
  }

  model = AE_model.AE_Net()
  checkpoint = torch.load(load_name)
  model.load_state_dict(checkpoint)
  model.cuda()
  model.eval()

  for inputs, targets in dataloaders['test']:
    # image_building = inputs[0,0,:,:]
    #
    # image_building[image_building == 0] = 2
    # image_building[image_building < 2] = 0
    # image_building[image_building == 2] = 1

    inputs = inputs.to(device)
    targets = targets.to(device)
    result = model(inputs)

    # result = result[0,2,:,:].detach().cpu().numpy()
    # # result = result * image_building.detach().cpu().numpy()
    # targets = targets[0,2,:,:].detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200)

    ax[0].set_title("(a) result", fontsize=10)
    ax[0].imshow(result[0,0,:,:].detach().cpu().numpy(),alpha=0.6,cmap=cm.coolwarm)

    ax[1].set_title("(a) building", fontsize=10)
    ax[1].imshow(targets[0,0,:,:].detach().cpu().numpy(),alpha=0.6,cmap=cm.coolwarm)

    # fig, ax = plt.subplots(nrows=3, ncols=2, dpi=200)

    # ax[0][0].set_title("(a) result", fontsize=10)
    # ax[0][0].imshow(result[0,2,:,:].detach().cpu().numpy(),alpha=0.6,cmap=cm.coolwarm)
    #
    # ax[0][1].set_title("(a) RSRP", fontsize=10)
    # ax[0][1].imshow(targets[0,2,:,:].detach().cpu().numpy(),alpha=0.6,cmap=cm.coolwarm)

    # ax[1][0].set_title("(a) result", fontsize=10)
    # ax[1][0].imshow(result[0, 0, :, :].detach().cpu().numpy(), alpha=0.6, cmap=cm.coolwarm)
    #
    # ax[1][1].set_title("(a) building", fontsize=10)
    # ax[1][1].imshow(targets[0,0,:,:].detach().cpu().numpy(), alpha=0.6, cmap=cm.coolwarm)
    #
    # ax[2][0].set_title("(a) result", fontsize=10)
    # ax[2][0].imshow(result[0, 1, :, :].detach().cpu().numpy(), alpha=0.6, cmap=cm.coolwarm)
    #
    # ax[2][1].set_title("(a) tx", fontsize=10)
    # ax[2][1].imshow(targets[0,1,:,:].detach().cpu().numpy(), alpha=0.6, cmap=cm.coolwarm)

    # ax.legend()
    plt.show()
    print('OK')
