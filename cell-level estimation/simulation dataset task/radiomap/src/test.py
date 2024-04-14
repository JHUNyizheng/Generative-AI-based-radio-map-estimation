from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import time
from opts import opts
import sys
from dataset.dataset_factory import get_dataset
from model.model import create_model, load_model, save_model
from model.loss import DiceBCELoss
import matplotlib.pyplot as plt
import cv2

def sample_mask(mask, sample_rate):
    id = np.transpose(np.nonzero(mask))
    id_idx = np.arange(len(id))
    random_id = np.random.choice(id_idx, int(len(id_idx) * sample_rate))

    b = np.zeros_like(mask)
    for sample_id in id[random_id]:
        b[sample_id[0],sample_id[1]] = 1
    return b

def RMSE(result,test_y):
    return np.sqrt(mean_squared_error(result, test_y))

def calculate_rmse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """ Prediction """
    score_RMSE = RMSE(y_pred, y_true)
    return score_RMSE

def calculate_nmse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    # y_true = y_true > 0.5
    #
    y_true = y_true.reshape(-1)

    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)

    """ Prediction """
    MSE = mean_squared_error(y_true, y_pred)
    NMSE = MSE / np.var(y_true)

    return NMSE

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_loss(opt,model):
    loss = []
    if opt.arch == 'RadioUnet':
        loss.append(nn.MSELoss())
    elif opt.arch == 'RadioCycle':
        loss.append(DiceBCELoss())
        loss.append(nn.L1Loss())
    elif opt.arch == 'RadioYnet':
        loss.append(DiceBCELoss())
        loss.append(nn.MSELoss())
    elif opt.arch == 'Interpolation':
        loss.append(nn.MSELoss())
    elif opt.arch == 'RadioTrans':
        loss.append(nn.MSELoss())
    elif opt.arch == 'RadioGan':
        loss.append(nn.MSELoss())
    return loss

""" modelG: rsrp + tx ---> building """
""" modelF: building + tx ---> rsrp """
def test(model,
          loader,
          loss,
          opt):
    rmse = 0
    nmse = 0
    min_commiuncation_quality = (105 - 47.84) / (186.41 - 47.84)
    cover = []
    if opt.arch == 'RadioUnet':
        epoch_loss_G = 0.0
        epoch_loss_F = 0.0
        modelF = model[0].eval()
        loss_F = loss[0]

        for build, radio, rsrp, tx in loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)

            tx = tx.to(opt.device, dtype=torch.float32)

            r_pred = modelF(torch.cat((build, tx), 1))

            cover_p = ((r_pred > min_commiuncation_quality) == True).sum(dim=(2, 3))
            cover_gt = ((radio > min_commiuncation_quality) == True).sum(dim=(2, 3))
            cover.append((cover_gt[0].item() - cover_p[0].item()) / cover_gt[0].item())

            lossF = loss_F(r_pred, radio)
            epoch_loss_F += lossF.item()

            rmse += calculate_rmse(r_pred, radio)
            nmse += calculate_nmse(r_pred, radio)


    elif opt.arch == 'RadioCycle':
        epoch_loss_G = 0.0
        epoch_loss_F = 0.0

        modelG = model[0].eval()
        modelF = model[1].eval()
        # loss_G = loss[0]
        # loss_F = loss[1]

        for build, radio, rsrp, tx in loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            b_pred = modelG(torch.cat((rsrp, tx), 1))
            # epoch_loss_G += loss_G(b_pred, build)

            r_pred = modelF(torch.cat((build, tx), 1))
            # epoch_loss_F += loss_F(r_pred, radio)
            cover_p = ((r_pred > min_commiuncation_quality) == True).sum(dim=(2, 3))
            cover_gt = ((radio > min_commiuncation_quality) == True).sum(dim=(2, 3))
            cover.append((cover_gt[0].item() - cover_p[0].item()) / cover_gt[0].item())

            rmse += calculate_rmse(r_pred, radio)
            nmse += calculate_nmse(r_pred, radio)

    elif opt.arch == 'RadioYnet':
        epoch_loss_G = 0.0
        epoch_loss_F = 0.0
        model = model[0].train()

        DiceBCELoss = loss[0]
        mseloss = nn.MSELoss()

        for build, radio, rsrp, tx in loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            y_pred, y_pred2 = model(torch.cat((rsrp, tx), 1))
            b_pred = torch.sigmoid(y_pred)
            b_pred = b_pred > 0.5

            plt.imshow(b_pred.detach().cpu().numpy()[0,0,:,:])
            plt.show()

            plt.imshow(build.detach().cpu().numpy()[0, 0, :, :])
            plt.show()

            lossG = DiceBCELoss(y_pred, build)
            lossF = mseloss(y_pred2, radio)

            epoch_loss_G += lossG.item()
            epoch_loss_F += lossF.item()

    elif opt.arch == 'RadioTrans':
        epoch_loss_G = 0.0
        epoch_loss_F = 0.0
        modelF = model[0].eval()
        loss_F = loss[0]

        for build, radio, rsrp, tx in loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            r_pred = modelF(torch.cat((build, tx), 1))

            cover_p = ((r_pred > min_commiuncation_quality) == True).sum(dim=(2, 3))
            cover_gt = ((radio > min_commiuncation_quality) == True).sum(dim=(2, 3))
            cover.append((cover_gt[0].item() - cover_p[0].item()) / cover_gt[0].item())

            lossF = loss_F(r_pred, radio)
            epoch_loss_F += lossF.item()

            rmse += calculate_rmse(r_pred, radio)
            nmse += calculate_nmse(r_pred, radio)

    elif opt.arch == 'RadioGan':
        epoch_loss_G = 0.0
        epoch_loss_F = 0.0
        Generator = model[0].eval()
        loss_F = loss[0]

        for build, radio, rsrp, tx in loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            r_pred = Generator(torch.cat((build, tx), 1))

            cover_p = ((r_pred > min_commiuncation_quality) == True).sum(dim=(2, 3))
            cover_gt = ((radio > min_commiuncation_quality) == True).sum(dim=(2, 3))
            cover.append((cover_gt[0].item() - cover_p[0].item()) / cover_gt[0].item())

            lossF = loss_F(r_pred, radio)
            epoch_loss_F += lossF.item()

            rmse += calculate_rmse(r_pred, radio)
            nmse += calculate_nmse(r_pred, radio)

    cover = np.array(cover)

    print("cover:", cover.mean())

    return rmse / len(loader), nmse / len(loader)

def main(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # opt.exp_id = "111"
    path = "/data/RadioUnet/"

    if opt.exp_id == 'default':
        print("exp_id null !!!")
        sys.exit(1)
    else:
        opt.model_path = os.path.join('..',"results", opt.arch, opt.exp_id)

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

    print('Creating model...')
    model = create_model(opt=opt)
    for i in range(len(model)):
        model[i] = model[i].to(opt.device)

    loss = get_loss(opt,model)

    if opt.load_model != '':
        model = load_model(model, opt.load_model, opt)
    else:
        model = load_model(model, opt.model_path, opt)

    """ testing the model """
    print('Starting testing...')

    start_time = time.time()
    rmse, nmse = test(model,test_loader,loss, opt)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    data_str = f'{opt.exp_id} | Testing Time: {epoch_mins}m {epoch_secs}s\n'
    data_str += f'\t{opt.exp_id} | Rmse Loss: {rmse:.4f}\t Nmse Loss: {nmse:.4f}\n'
    print(data_str)

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)