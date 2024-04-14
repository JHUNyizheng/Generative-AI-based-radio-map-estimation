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
def test(model,test_loader,opt,rate):
    if opt.arch == 'Interpolation':
        modelG = model[0].eval()
        test_loader.dataset.SampleRate(rate)
        rmse = 0
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        for build, radio, rsrp, tx in test_loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            r_pred = modelG(torch.cat((rsrp, tx), 1))
            rmse += calculate_rmse(rsrp,r_pred)
        rmse = rmse / len(test_loader)

    if opt.arch == 'RadioUnet':
        modelF = model[0].eval()
        test_loader.dataset.SampleRate(rate)
        rmse = 0
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        for build, radio, rsrp, tx in test_loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            r_pred = modelF(torch.cat((build, tx), 1))
            rmse += calculate_rmse(rsrp,r_pred)
        rmse = rmse / len(test_loader)

    elif opt.arch == 'RadioCycle':
        modelG = model[0].eval()
        modelF = model[1].eval()
        test_loader.dataset.SampleRate(rate)
        rmse = 0
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for build, radio, rsrp, tx in test_loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            b_pred = modelG(torch.cat((rsrp, tx), 1))

            # sigmid转换为概率分布
            fake_B = torch.sigmoid(b_pred)
            b_pred = torch.zeros_like(b_pred)
            b_pred[fake_B > 0.5] = 1
            b_pred[fake_B <= 0.5] = 0
            r_pred = modelF(torch.cat((b_pred, tx), 1))

            rmse += calculate_rmse(radio,r_pred)
            score = calculate_metrics(build, fake_B)
            metrics_score = list(map(add, metrics_score, score))
        rmse = rmse / len(test_loader)
        metrics_score = [metrics / len(test_loader) for metrics in metrics_score]

    elif 'RadioYnet' in opt.arch:
        modelG = model[0].eval()
        test_loader.dataset.SampleRate(rate)
        rmse = 0
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for build, radio, rsrp, tx in test_loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)


            b_pred, r_pred = modelG(torch.cat((rsrp, tx), 1))
            b_pred = torch.sigmoid(b_pred)

            if 'NRM' in opt.arch:
                score = calculate_metrics(build, b_pred)
                metrics_score = list(map(add, metrics_score, score))
            else:
                rmse += calculate_rmse(radio, r_pred)
                score = calculate_metrics(build, b_pred)
                metrics_score = list(map(add, metrics_score, score))

            # rmse += calculate_rmse(radio, r_pred)
            # score = calculate_metrics(build, b_pred)
            # metrics_score = list(map(add, metrics_score, score))
            # print("Rmse", rmse)
        rmse = rmse / len(test_loader)
        metrics_score = [metrics / len(test_loader) for metrics in metrics_score]

    return rmse, metrics_score

if __name__ == '__main__':
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # opt.exp_id = "cycleDWA_s20"
    opt.save_log = 'True'
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
        batch_size=6,
        shuffle=True,
        num_workers=opt.num_workers
    )

    """ testing the model """
    print('Starting testing...')

    per_step = 5
    per1 = 0
    per2 = 50

    Jaccard, F1, Recall, Precision, Acc, AP = [], [], [], [], [], []
    Rmse = []
    Rate = []

    for i in range(int((per2 - per1) / per_step)):
        rate = per2 - i * per_step
        rmse, metrics_score = test(model,test_loader, opt,rate)

        Jaccard.append(metrics_score[0])
        F1.append(metrics_score[1])
        Recall.append(metrics_score[2])
        Precision.append(metrics_score[3])
        Acc.append(metrics_score[4])
        AP.append(metrics_score[5])
        Rmse.append(rmse)
        Rate.append(rate)
        print(per2 - i * per_step)

    print("Jaccard",Jaccard)
    print("F1", F1)
    print("Recall", Recall)
    print("Precision", Precision)
    print("Acc", Acc)
    print("Acc", AP)
    print("Rmse", Rmse)
    print("Rate", Rate)

    if opt.save_log == 'True':
        with open(os.path.join(opt.model_path + 'statistics.txt'), "a") as log_file:
            log_file.truncate(0)
            now = time.strftime("%c")
            log_file.write('================ test %s================\n' % now)
            log_file.write('Jaccard : \n%s\n' % (Jaccard))  # save the message
            log_file.write('F1 : \n%s\n' % (F1))  # save the message
            log_file.write('Recall : \n%s\n' % (Recall))  # save the message
            log_file.write('Precision : \n%s\n' % (Precision))  # save the message
            log_file.write('Acc : \n%s\n' % (Acc))  # save the message
            log_file.write('AP : \n%s\n' % (AP))  # save the message
            log_file.write('RMSE : \n%s\n' % (Rmse))  # save the message
            log_file.write('sample_ratio : \n%s' % (Rate))  # save the message

