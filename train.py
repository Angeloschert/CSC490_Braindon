# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
# import tensorflow as tf
# from niftynet.layer.loss_segmentation import LossFunction
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet

from torchgeometry.losses.dice import *
import torch
import torch.nn as nn


class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()

def train(config_file):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
     
    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)
   
    # 2, construct graph
    full_data_shape  = [batch_size] + config_data['data_shape']
    full_label_shape = [batch_size] + config_data['label_shape']
    x = torch.zeros(full_data_shape, dtype=torch.float32, requires_grad=True)
    y = torch.zeros(full_label_shape, dtype=torch.int64, requires_grad=True)
    w = torch.zeros(full_label_shape, dtype=torch.float32, requires_grad=True)
   
    w_regularizer = config_train.get('decay', 1e-7)
    b_regularizer = config_train.get('decay', 1e-7)
    net_class = NetFactory.create(net_type)
    net = net_class(in_chns = 5, # not sure
                    num_classes = class_num,
                    w_reg = w_regularizer,
                    b_reg = b_regularizer,
                    name = net_name)
    predicty = net(x)
    proby = torch.softmax(predicty)
    
    loss = DiceLoss()
    print('size of predicty:',predicty)
    
    # 3, initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt = torch.optim.Adam(w, lr=lr)
    # saver = tf.train.Saver()
    
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    
    # 4, start to train
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it  = config_train.get('start_iteration', 0)

    if(start_it > 0):
        checkpoint = torch.load(config_train['model_pre_trained'])
        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_list, temp_loss_list = [], []
    for n in range(start_it, config_train['maximal_iteration']):
        train_pair = dataloader.get_subimage_batch()
        tempx = train_pair['images']
        # tempw = train_pair['weights']
        tempy = train_pair['labels']

        opt.zero_grad()
        pred = net(tempx)
        dice_loss = loss(pred, tempy)
        dice_loss.backward()
        opt.step()

        if(n % config_train['test_iteration'] == 0):
            batch_dice_list = []
            for step in range(config_train['test_step']):
                train_pair = dataloader.get_subimage_batch()
                tempx = train_pair['images']
                # tempw = train_pair['weights']
                tempy = train_pair['labels']
                dice_loss = loss(tempx, tempy)
                batch_dice_list.append(dice_loss)
            batch_dice = np.asarray(batch_dice_list, np.float32).mean()

            t = time.strftime('%X %x %Z')
            print(t, 'n', n,'loss', batch_dice)
            loss_list.append(batch_dice)
            np.savetxt(loss_file, np.asarray(loss_list))

        if((n+1) % config_train['snapshot_iteration']  == 0):
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': dice_loss,
            }, config_train['model_save_prefix']+"_{0:}.pt".format(n+1))
            # saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/train_wt_ax.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    train(config_file)