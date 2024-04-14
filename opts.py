from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json
import torch

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--dataset', default='simulation',
                                 help='real | simulation')
        self.parser.add_argument('--test_dataset', default='',
                                 help='real | simulation')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--no_pause', action='store_true')
        self.parser.add_argument('--demo', default='',
                                 help='path to image/ image folders/ video. '
                                      'or "webcam"')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')

        # system
        self.parser.add_argument('--gpus', default='1',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet
        self.parser.add_argument('--not_set_cuda_env', action='store_true',
                                 help='used when training in slurm clusters.')

        # model
        self.parser.add_argument('--arch', default='RadioTrans',
                                 help='knn | Randomforest | Interpolation | RadioUnet | RadioCycle | RadioYnet | RadioTrans')

        # input
        self.parser.add_argument('--input_h', type=int, default=256,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=256,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--optim', default='adam')
        self.parser.add_argument('--lr', type=float, default=2e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='60',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--save_point', type=str, default='90',
                                 help='when to save the model to disk.')
        self.parser.add_argument('--num_epochs', type=int, default=50,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help='batch size')
        self.parser.add_argument('--loss_switch', default='DWA',
                                 help='Only RadioCycle Can Use.choose None | DWA')

        # test
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--save_log', default='False',
                                 help='Save or not')


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        if opt.test_dataset == '':
            opt.test_dataset = opt.dataset

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        # opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.save_point = [int(i) for i in opt.save_point.split(',')]
        opt.num_workers = max(opt.num_workers, 2 * len(opt.gpus))
        opt.pre_img = False


        # # log dirs
        # opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        # opt.data_dir = os.path.join(opt.root_dir, 'data')
        # opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        # opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        # opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        #
        # if opt.resume and opt.load_model == '':
        #     opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
        return opt


    def init(self, args=''):
        print("OK")

