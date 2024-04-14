from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from .network.RadioCycle import RadioCycleG,RadioCycleF
from .network.RadioUnet import RadioUNet
from .network.RadioYnet import RadioYnet
from .network.Interpolation import RadioWNet
from .network.RadioTrans import RadioTrans
from .network.RadioGan import Generator, Discriminator


_network_factory = {
    # rsrp ---> rsrp
    'knn': [KNeighborsRegressor],
    'Randomforest': [RandomForestRegressor],
    'Interpolation': [RadioWNet()],

    # building ---> rsrp
    'RadioUnet': [RadioUNet()],

    # rsrp ---> building ---> rsrp
    'RadioCycle': [RadioCycleG(), RadioCycleF()],
    'RadioYnet':[RadioYnet(None)],
    'RadioYnet_ECA': [RadioYnet('ECA')],
    'RadioYnet_DA': [RadioYnet('DA')],
    'RadioYnet_CA': [RadioYnet('CA')],
    'RadioYnet_NRM': [RadioYnet('NRM')],
    'RadioTrans':[RadioTrans()],
    'RadioGan': [Generator(), Discriminator()],
}

def create_model(opt=None):
    modellist = _network_factory[opt.arch]
    return modellist

def load_model(model, model_path, opt):
    if opt.arch == 'RadioUnet':
        model_nameF = "checkpoint_F.pth"
        model[0].load_state_dict(torch.load(os.path.join(model_path, model_nameF), map_location=opt.device))
    elif opt.arch == 'RadioCycle':
        model_nameG = "checkpoint_G.pth"
        model_nameF = "checkpoint_F.pth"
        model[0].load_state_dict(torch.load(os.path.join(model_path, model_nameG), map_location=opt.device))
        model[1].load_state_dict(torch.load(os.path.join(model_path, model_nameF), map_location=opt.device))
    elif 'RadioYnet' in opt.arch:
        model_nameG = "checkpoint_G.pth"
        model[0].load_state_dict(torch.load(os.path.join(model_path, model_nameG), map_location=opt.device))
    elif opt.arch == 'interpolation':
        model_nameG = "checkpoint_G.pth"
        model[0].load_state_dict(torch.load(os.path.join(model_path, model_nameG), map_location=opt.device))
    elif opt.arch == 'RadioTrans':
        model_nameF = "checkpoint_F.pth"
        model[0].load_state_dict(torch.load(os.path.join(model_path, model_nameF), map_location=opt.device))
    elif opt.arch == 'RadioGan':
        model_nameG = "checkpoint_G.pth"
        model_nameD = "checkpoint_D.pth"
        model[0].load_state_dict(torch.load(os.path.join(model_path, model_nameG), map_location=opt.device))
        model[1].load_state_dict(torch.load(os.path.join(model_path, model_nameD), map_location=opt.device))
    return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

