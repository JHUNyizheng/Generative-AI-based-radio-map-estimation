from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data
import torch.nn as nn
from skimage import io
import cv2
import numpy as np
from operator import add
from torch.utils.data import DataLoader
import sys
import time
sys.path.append("../..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../../") for name in dirs])
from src.opts import opts
from src.dataset.dataset_factory import get_dataset
from src.model.model import create_model, load_model, save_model
from src.model.utils import calculate_metrics, calculate_rmse


""" modelG: rsrp + tx ---> building """
""" modelF: building + tx ---> rsrp """
def make_dataset(model,test_loader,opt):
    if opt.arch == 'RadioUnet':
        modelF = model[0].eval()
        map = torch.Tensor([256, 256])
        h, w = 256, 256
        for build, radio, rsrp, tx in test_loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    tx = torch.zeros_like(tx).to(opt.device, dtype=torch.float32)
                    tx[0, 0, i - 1, j] = 1.0
                    tx[0, 0, i + 1, j] = 1.0
                    tx[0, 0, i, j] = 1.0
                    tx[0, 0, i, j + 1] = 1.0
                    tx[0, 0, i, j - 1] = 1.0
                    r_pred = modelF(torch.cat((build, tx), 1))
                    print('OK')



    return rmse, metrics_score

if __name__ == '__main__':
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    opt.exp_id = "cycleDWA_s5"

    path = "/data/RadioUnet/"
    if opt.exp_id == 'default':
        print("exp_id null !!!")
        sys.exit(1)
    else:
        opt.model_path = os.path.join('..','..', "results", opt.arch, opt.exp_id)

    """ Load Model """
    print('Creating model...')
    model = create_model(opt=opt)
    for i in range(len(model)):
        model[i] = model[i].to(opt.device)

    if opt.load_model != '':
        model = load_model(model, opt.load_model, opt)
    else:
        model = load_model(model, opt.model_path, opt)

    """ Dataset and loader """
    Dataset = get_dataset(opt.dataset)
    test_dataset = Dataset(path, 'test')

    print("loading trainset...")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=opt.num_workers
    )

    """ testing the model """
    print('Starting testing...')


    rmse, metrics_score = make_dataset(model,test_loader, opt)






