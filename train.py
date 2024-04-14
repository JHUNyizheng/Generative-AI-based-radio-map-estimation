from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import sys
import torch.utils.data
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
from opts import opts
from dataset.dataset_factory import get_dataset
from model.model import create_model, load_model, save_model
from model.loss import DiceBCELoss, Generator_loss, Discriminator_loss
from model.utils import get_scheduler

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_optimizer(opt, model):
    optimizer = []
    if opt.optim == 'adam':
        for i in range(len(model)):
            optimizer.append(torch.optim.Adam(model[i].parameters(), opt.lr))
    elif opt.optim == 'sgd':
        for i in range(len(model)):
            optimizer.append(torch.optim.SGD(model[i].parameters(), opt.lr, momentum=0.9, weight_decay=0.0001))
    return optimizer

def get_loss(opt,model):
    loss = []
    if opt.arch == 'RadioUnet':
        loss.append(nn.MSELoss())
    elif opt.arch == 'RadioCycle':
        loss.append(DiceBCELoss())
        loss.append(nn.MSELoss())
    elif 'RadioYnet' in opt.arch:
        loss.append(DiceBCELoss())
        loss.append(nn.MSELoss())
    elif opt.arch == 'Interpolation':
        loss.append(nn.MSELoss())
    elif opt.arch == 'RadioTrans':
        loss.append(nn.MSELoss())
    elif opt.arch == 'RadioGan':
        loss.append(Generator_loss())
        loss.append(Discriminator_loss())
    return loss


def evaluate1(model, loader, loss_fn, opt):
    epoch_loss1 = 0.0
    epoch_loss2 = 0.0
    mseloss = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        for build, radio, rsrp, tx in loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            y_pred, y_pred2 = model(torch.cat((rsrp, tx), 1))
            if 'NRM' in opt.arch:
                loss1 = loss_fn(y_pred, build)
                epoch_loss1 += loss1.item()
            else:
                loss1 = loss_fn(y_pred, build)
                loss2 = mseloss(y_pred2, radio)
                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()

        epoch_loss1 = epoch_loss1 / len(loader)
        epoch_loss2 = epoch_loss2 / len(loader)
    return epoch_loss1, epoch_loss2

def evaluate(model, loader, loss_fn, opt, model_opt='G'):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for build, radio, rsrp, tx in loader:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            if model_opt == 'G':
                x = torch.cat((rsrp, tx), 1)
                y = build
            elif model_opt == 'F':
                x = torch.cat((build, tx), 1)
                y = radio
            elif model_opt == 'I':
                x = torch.cat((rsrp, tx), 1)
                y = radio

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

""" modelG: rsrp + tx ---> building """
""" modelF: building + tx ---> rsrp """
def train(model,
          loader,
          optimizer,
          scheduler,
          loss,
          best_valid_loss,
          epoch,
          opt):
    if opt.arch == 'Interpolation':
        epoch_loss_G = 0.0
        best_valid_lossG = 10.0
        best_valid_lossF = 0.0
        modelG = model[0].train()
        optimizer = optimizer[0]
        loss_G = loss[0]
        step = 0

        for build, radio, rsrp, tx in loader[0]:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            optimizer.zero_grad()
            r_pred = modelG(torch.cat((rsrp, tx), 1))
            lossG = loss_G(r_pred, radio)

            lossG.backward()
            optimizer.step()

            epoch_loss_G += lossG.item()
            step += 1
            # print(step)
            if step % 200 == 0:
                valid_loss = evaluate(modelG, loader[1], loss_G, opt,model_opt='I')
                print(step, "train step_loss:", epoch_loss_G / step, "val step_loss:", valid_loss)

                """ Saving the model """
                if valid_loss < best_valid_loss:
                    data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}."
                    data_str += f'\t Val. LossG: {valid_loss:.3f}\t '
                    print(data_str)
                    best_valid_loss = valid_loss
                    torch.save(modelG.state_dict(), os.path.join(opt.model_path,"checkpoint_G.pth"))
    if opt.arch == 'RadioUnet':
        epoch_loss_G = 0.0
        best_valid_lossG = 10.0
        best_valid_lossF = 0.0
        modelF = model[0].train()
        optimizer = optimizer[0]
        loss_F = loss[0]
        step = 0

        for build, radio, rsrp, tx in loader[0]:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            optimizer.zero_grad()
            r_pred = modelF(torch.cat((build, tx), 1))

            build[build != 0] = 1
            build_mask = (build - 1).cuda(non_blocking=True)
            build_mask[build_mask == -1] = 1

            lossG = loss_F(r_pred, radio)

            lossG.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss_G += lossG.item()
            step += 1
            # print(step)
            if step % 200 == 0:
                valid_loss = evaluate(modelF, loader[1], loss_F, opt, model_opt='F')
                print(step, "train step_loss:", epoch_loss_G / step, "val step_loss:", valid_loss)

                data_str = f'{epoch} | {step} | train_step_loss: {epoch_loss_G / step:.4f} | val_step_loss: {valid_loss:.4f}'
                with open(opt.model_path + '/Train_log.txt', 'a') as f:
                    # f.write(print_str+'\t')
                    f.write(data_str + '\n')

                """ Saving the model """
                if valid_loss < best_valid_loss:
                    data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}."
                    data_str += f'\t Val. LossG: {valid_loss:.3f}\t '
                    print(data_str)
                    best_valid_loss = valid_loss
                    torch.save(modelF.state_dict(), os.path.join(opt.model_path,"checkpoint_F.pth"))

    elif opt.arch == 'RadioCycle':
        epoch_loss_G = 0.0
        epoch_loss_F = 0.0
        epoch_loss_Gf = 0.0
        epoch_loss_Ff = 0.0
        epoch_loss_Gr = 0.0
        epoch_loss_Fr = 0.0

        best_valid_lossG = 10.0
        best_valid_lossF = 10.0
        lamb = 20
        modelG = model[0].train()
        modelF = model[1].train()
        optimizerG = optimizer[0]
        optimizerF = optimizer[1]
        loss_G = loss[0]
        loss_F = loss[1]

        step = 0
        cnt = 0
        for build, radio, rsrp, tx in loader[0]:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            optimizerG.zero_grad()
            b_pred = modelG(torch.cat((rsrp, tx), 1))
            lossG = loss_G(b_pred, build)

            optimizerF.zero_grad()
            r_pred = modelF(torch.cat((build, tx), 1))
            lossF = loss_F(r_pred, radio)

            # sigmid转换为概率分布
            fake_B = torch.sigmoid(b_pred)
            b_pred = torch.zeros_like(b_pred)

            b_pred[fake_B > 0.5] = 1
            b_pred[fake_B <= 0.5] = 0

            b_rec = modelG(torch.cat((r_pred, tx), 1))
            lossG_f = loss_G(b_rec, build)

            r_rec = modelF(torch.cat((b_pred, tx), 1))
            lossF_f = loss_F(r_rec, radio)

            lossG_r = loss_G(b_rec, b_pred)
            lossF_r = loss_F(r_rec, r_pred)

            loss_G_total = lossG + lossG_f + lossG_r
            loss_F_total = lamb * (lossF + lossF_f + lossF_r)

            t_loss = loss_G_total * opt.lambda_weight[0] + loss_F_total * opt.lambda_weight[1]
            t_loss.backward()
            optimizerG.step()
            optimizerF.step()
            scheduler.step()

            epoch_loss_G += lossG.item()
            epoch_loss_F += lossF.item()
            epoch_loss_Gf += lossG_f.item()
            epoch_loss_Ff += lossF_f.item()
            epoch_loss_Gr += lossG_r.item()
            epoch_loss_Fr += lossF_r.item()
            step += 1
            # print(step)
            if step % 200 == 0:
                cnt += 1
                if cnt <= 3:
                    opt.lambda_weight[0] = 1
                    opt.lambda_weight[1] = 1
                else:
                    w_1 = opt.avg_cost[1, 0] / opt.avg_cost[0, 0]
                    w_2 = opt.avg_cost[1, 1] / opt.avg_cost[0, 1]
                    opt.lambda_weight[0] = 2 * np.exp(w_1 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2))
                    opt.lambda_weight[1] = 2 * np.exp(w_2 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2))

                # lamb = 20
                # t_loss = lossG + lossG_f + lossG_r + lamb * (lossF + lossF_f + lossF_r)
                # t_loss = loss_G_total * lambda_weight[0] + loss_F_total * lambda_weight[1]

                opt.avg_cost[0, :] = opt.avg_cost[1, :]
                opt.avg_cost[1, :] = opt.avg_cost[2, :]

                opt.avg_cost[2, 0] = loss_G_total
                opt.avg_cost[2, 1] = loss_F_total

                print_str = f'{epoch}|{cnt}|lambda_weight:||{opt.lambda_weight[0]:2.5f}|{opt.lambda_weight[1]:2.6f}||'
                print(print_str)

                valid_lossG = evaluate(modelG, loader[1], loss_G, opt, model_opt='G')
                valid_lossF = evaluate(modelF, loader[1], loss_F, opt, model_opt='F')
                print_str = f'{epoch}|{step}|train_GF_GFf_GFr:||{epoch_loss_G / step:2.5f}|{epoch_loss_F / step:2.6f}||' \
                            f'{epoch_loss_Gf / step:2.5f}|{epoch_loss_Ff / step:2.6f}||' \
                            f'{epoch_loss_Gr / step:2.5f}|{epoch_loss_Fr / step:2.6f}||' \
                            f'val_GF:||{valid_lossG:2.5f}|{valid_lossF:2.5f}||'
                print(print_str)

                data_str = f'{epoch} | {step} | train_step_loss: {epoch_loss_F / step:.4f} | val_step_loss: {valid_lossF:.4f}'
                with open(opt.model_path + '/Train_log.txt', 'a') as f:
                    # f.write(print_str+'\t')
                    f.write(data_str + '\n')

                if opt.dataset == 'radiomap3d':
                    valid_loss = valid_lossF
                else:
                    valid_loss = valid_lossG + lamb * valid_lossF

                """ Saving the model """
                if valid_loss < best_valid_loss:
                    data_str = f"Valid loss improved from {best_valid_loss:2.6f} to |{valid_loss:2.6f}|"
                    data_str += f'\t Val. LossG: {valid_lossG:.6f}\t  Val. LossF: {valid_lossF:.6f}\n'

                    print(data_str)
                    best_valid_loss = valid_loss
                    best_valid_lossG = valid_lossG
                    best_valid_lossF = valid_lossF

                    torch.save(modelG.state_dict(), os.path.join(opt.model_path,"checkpoint_G.pth"))
                    torch.save(modelF.state_dict(), os.path.join(opt.model_path,"checkpoint_F.pth"))

    elif 'RadioYnet' in opt.arch:
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        best_valid_lossG = 0.0
        best_valid_lossF = 0.0
        model = model[0].train()
        step = 0
        DiceBCELoss = loss[0]
        mseloss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

        for build, radio, rsrp, tx in loader[0]:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred, y_pred2 = model(torch.cat((rsrp, tx), 1))
            if 'NRM' in opt.arch:
                loss1 = DiceBCELoss(y_pred, build)
                loss = loss1
                epoch_loss1 += loss1.item()
            else:
                loss1 = DiceBCELoss(y_pred, build)
                loss2 = mseloss(y_pred2, radio)
                loss = loss1 + loss2
                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()

            loss.backward()
            optimizer.step()
            # scheduler.step()

            step += 1
            # print(step)
            if step % 500 == 0:
                print(step, "train step_loss:", epoch_loss1 / step, epoch_loss2 / step)
                valid_loss1, valid_loss2 = evaluate1(model, loader[1], DiceBCELoss, opt)

                """ Saving the model """
                if valid_loss1 < best_valid_loss:
                    data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss1:2.4f}"
                    data_str += f'\t Val. Loss1: {valid_loss1:.3f} \t Val. Loss2: {valid_loss2:.5f}\n'
                    print(data_str)
                    best_valid_loss = valid_loss1
                    best_valid_lossG = valid_loss1
                    best_valid_lossF = valid_loss2
                    torch.save(model.state_dict(), os.path.join(opt.model_path,"checkpoint_G.pth"))

                torch.save(model.state_dict(), os.path.join(opt.model_path, "checkpoint_G_last.pth"))
        epoch_loss1 = epoch_loss1 / len(loader)
    elif opt.arch == 'RadioTrans':
        epoch_loss_G = 0.0
        best_valid_lossG = 10.0
        best_valid_lossF = 0.0
        modelF = model[0].train()
        optimizer = optimizer[0]
        loss_F = loss[0]
        step = 0

        for build, radio, rsrp, tx in loader[0]:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            optimizer.zero_grad()
            r_pred = modelF(torch.cat((build, tx), 1))
            lossG = loss_F(r_pred, radio)

            lossG.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss_G += lossG.item()
            step += 1
            # print(step)
            if step % 200 == 0:
                valid_loss = evaluate(modelF, loader[1], loss_F, opt, model_opt='F')
                print(step, "train step_loss:", epoch_loss_G / step, "val step_loss:", valid_loss)

                data_str = f'{epoch} | {step} | train_step_loss: {epoch_loss_G / step:.4f} | val_step_loss: {valid_loss:.4f}'
                with open(opt.model_path + '/Train_log.txt', 'a') as f:
                    # f.write(print_str+'\t')
                    f.write(data_str + '\n')

                """ Saving the model """
                if valid_loss < best_valid_loss:
                    data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}."
                    data_str += f'\t Val. LossG: {valid_loss:.3f}\t '
                    print(data_str)
                    best_valid_loss = valid_loss
                    torch.save(modelF.state_dict(), os.path.join(opt.model_path, "checkpoint_F.pth"))


    elif opt.arch == 'RadioGan':
        epoch_loss_G = 0.0
        best_valid_lossG = 10.0
        best_valid_lossF = 0.0
        Generator = model[0].train()
        Discriminator = model[1].train()
        g_optimizer = optimizer[0]
        d_optimizer = optimizer[1]
        loss_G = loss[0]
        loss_D = loss[1]
        step = 0

        for build, radio, rsrp, tx in loader[0]:
            build = build.to(opt.device, dtype=torch.float32)
            radio = radio.to(opt.device, dtype=torch.float32)
            rsrp = rsrp.to(opt.device, dtype=torch.float32)
            tx = tx.to(opt.device, dtype=torch.float32)

            # with torch.autograd.detect_anomaly():

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            gen_output = Generator(torch.cat((build, tx), 1))

            disc_real_output = Discriminator([build, radio])
            disc_generated_output = Discriminator([build, gen_output])

            gen_loss = loss_G(disc_generated_output, gen_output, radio)
            disc_loss = loss_D(disc_real_output, disc_generated_output)
            # lossG = loss_F(r_pred, radio)
            # torch.autograd.set_detect_anomaly(True)

            gen_loss.backward(retain_graph=True)
            disc_loss.backward(retain_graph=True)

            g_optimizer.step()
            d_optimizer.step()
            scheduler.step()

            epoch_loss_G += gen_loss.item()
            step += 1
            # print(step)
            if step % 200 == 0:
                valid_loss = evaluate(Generator, loader[1], nn.MSELoss(), opt, model_opt='F')
                print(step, "train step_loss:", epoch_loss_G / step, "val step_loss:", valid_loss)
                # print(step, "train step_loss:", epoch_loss_G / step, "val step_loss:", 0)

                data_str = f'{epoch} | {step} | train_step_loss: {epoch_loss_G / step:.4f} | val_step_loss: {valid_loss:.4f}'
                with open(opt.model_path + '/Train_log.txt', 'a') as f:
                    # f.write(print_str+'\t')
                    f.write(data_str + '\n')

                """ Saving the model """
                if valid_loss < best_valid_loss:
                    data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}."
                    data_str += f'\t Val. LossG: {valid_loss:.3f}\t '
                    print(data_str)
                    best_valid_loss = valid_loss
                    torch.save(Generator.state_dict(), os.path.join(opt.model_path, "checkpoint_G.pth"))
                    torch.save(Discriminator.state_dict(), os.path.join(opt.model_path, "checkpoint_D.pth"))


    return best_valid_lossG, best_valid_lossF, best_valid_loss

def main(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # opt.exp_id = "cycleDWA_s5"

    """ Create a directory. """
    if opt.exp_id == 'default':
        print("exp_id null !!!")
        sys.exit(1)
    else:
        opt.model_path = os.path.join('..',"results", opt.arch, opt.exp_id)

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    """ Dataset and loader """
    path = "/data/RadioUnet/"
    Dataset = get_dataset(opt.dataset)
    train_dataset = Dataset(path, 'train')
    valid_dataset = Dataset(path, 'val')

    print("loading trainset...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )
    print("loading valset...")
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    loader = [train_loader,valid_loader]

    print('Creating model...')
    model = create_model(opt=opt)
    for i in range(len(model)):
        model[i] = model[i].to(opt.device)

    loss = get_loss(opt,model)
    optimizer = get_optimizer(opt, model)
    scheduler = get_scheduler(optimizer, len(train_loader), opt)
    start_epoch = 0

    if opt.load_model != '':
        model = load_model(model, opt.load_model, opt)

    """ Training the model """

    print('Starting training...')
    if opt.loss_switch == 'DWA':
        opt.avg_cost = np.zeros([3, 2], dtype=np.float32)
        opt.lambda_weight = np.ones([2])

    best_valid_loss = float("inf")
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print("start...")
        start_time = time.time()
        best_vaild_lossG, best_vaild_lossF,best_valid_loss = train(model, loader, optimizer, scheduler, loss, best_valid_loss, epoch, opt)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'{opt.exp_id} | {opt.arch} | Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\t{opt.exp_id} | {opt.arch} | Val LossG: {best_vaild_lossG:.4f}\t Best Valid Loss: {best_valid_loss:.4f}\n'
        # data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)

