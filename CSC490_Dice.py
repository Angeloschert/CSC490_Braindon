#!/usr/bin/env python
# coding: utf-8

# In[7]:



from __future__ import absolute_import, print_function

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet
from util.diceloss import *
# https://github.com/JunMa11/SegLoss
from util.SegLoss.losses_pytorch.dice_loss import *

# from torchgeometry.losses.dice import *
import torch
import torch.nn as nn
# from kornia.losses import DiceLoss as KDiceLoss
import warnings
warnings.filterwarnings('ignore')


# In[8]:




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()
        
def diceProprocess(pred, tempy):
    B, Mp, H, W, Sp = pred.shape
    _, _, _, _, Sy = tempy.shape
    global device
    pred_ = torch.zeros([B, Mp, H, W], dtype=torch.float32, requires_grad=True).to(device)
    tempy_ = torch.zeros([B, Mp, H, W], dtype=torch.float32, requires_grad=True).to(device)

    for i in range(B):
      for sy in range(Sy):
        tempy_.data[i, 0] += tempy[i, 0, :, :, sy]
      tempy_.data[i, 1] = tempy_.data[i, 0]
      for mp in range(Mp):
        for sp in range(Sp):
          pred_.data[i, mp] += pred[i, mp, :, :, sp]

    return pred_, tempy_


# In[ ]:
def Dice(inp, target, eps=1):
	# 抹平了，弄成一维的
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)
    # 返回值，让值在0和1之间波动
    return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)




# In[9]:


config_file = './config17/train_wt_ax_local.txt'
# config_file = './config17/train_wt_ax_local.txt'


# In[10]:


# 1, load configuration parameters
config = parse_config(config_file)
config_data = config['data']
config_net = config['network']
config_train = config['training']

random.seed(config_train.get('random_seed', 1))
assert (config_data['with_ground_truth'])

net_type = config_net['net_type']
net_name = config_net['net_name']
class_num = config_net['class_num']
batch_size = config_data.get('batch_size', 5)

# 2, construct graph
full_data_shape = [batch_size] + config_data['data_shape']
full_label_shape = [batch_size] + config_data['label_shape']
x = torch.zeros(full_data_shape, dtype=torch.float32, requires_grad=True)
w = torch.zeros(full_label_shape, dtype=torch.float32, requires_grad=True)

w_regularizer = config_train.get('decay', 1e-7)
b_regularizer = config_train.get('decay', 1e-7)
print(w_regularizer)

net_class = NetFactory.create(net_type)
net = net_class(
    in_chns=full_data_shape[1],  # not sure
    num_classes=class_num,
    w_reg=w_regularizer,
    b_reg=b_regularizer
)

net = net.to(device)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('PReLU') != -1:
        m.inplace = True


net.apply(inplace_relu)
count = 0
for m in net.modules():
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.bias)
        
        count += 1
print('=======  ============', count, '=====================')

# print('\nSize of x:', x.shape)
# print('Size of predicty:', predicty.shape)
# print('Size of proby:', proby.shape)

# 3, initialize optimizer
lr = config_train.get('learning_rate', 1)
opt = torch.optim.Adam(net.parameters(), lr=lr)
# dice_loss = DiceLoss().to(device)
dice_loss = GDiceLoss() # use v1 diceloss
# dice_loss = KDiceLoss().to(device)
ce_loss = nn.CrossEntropyLoss().to(device)

dataloader = DataLoader(config_data)
dataloader.load_data()

# 4, start to train
loss_file = config_train['model_save_prefix'] + "_loss.txt"
start_it = config_train.get('start_iteration', 0)

if (start_it > 0):
    checkpoint = torch.load(config_train['model_pre_trained'])
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

loss_list, temp_loss_list = [], []
for n in range(start_it, config_train['maximal_iteration']):
    # print("Inside main loop: " + str(n))
    train_pair = dataloader.get_subimage_batch()
    tempx = torch.permute(torch.from_numpy(train_pair['images']), [0, 4, 3,2 ,1]).to(device)
    tempy = torch.permute(torch.from_numpy(train_pair['labels']), [0, 4, 3,2 ,1]).to(device)
    pred = net(tempx)
    # pred = torch.max(pred, dim=1).indices
    # print(tempy.shape)
    # print(pred.shape)
    # print(pred.shape)
    # print(torch.unique(pred))
    # print("\ntrain time net pass")

    # pred, tempy = diceProprocess(pred, tempy)

    # print("\nInside train.py - testing iter, tempx shape: " + str(tempx.shape))
    # print("Inside train.py - testing iter, tempy shape: " + str(tempy.shape))
    # print("Inside train.py - testing iter, pred shape: " + str(pred.shape))

    # proby = torch.softmax(pred)

    # loss = dice_loss(pred, tempy)
    tempy = tempy.reshape((-1, 144, 144, 11))
    loss = dice_loss(pred, tempy) # 尝试使用dice loss
    # loss = dice_loss(pred, tempy)
    # loss = dice_loss(pred, tempy)
    opt.zero_grad()
    
    # print("\ntrain time dice loss pass")

    loss.backward()
    opt.step()

    if (n % config_train['test_iteration'] == 0):
        batch_dice_list = []
        dice_score = 0.0
        with torch.no_grad():
            for step in range(config_train['test_step']):
                # print("\nInside testing loop: " + str(step))
                train_pair = dataloader.get_subimage_batch()
                tempx = torch.permute(torch.from_numpy(train_pair['images']), [0, 4, 3, 2 ,1]).to(device)
                tempy = torch.permute(torch.from_numpy(train_pair['labels']), [0, 4, 3, 2 ,1]).to(device)
                pred = net(tempx)
                tempy = tempy.reshape((-1, 144, 144, 11))
                loss = dice_loss(pred, tempy)
                pred = torch.max(pred, dim=1).indices
                # dice_score += Dice(pred.cpu().numpy(), tempy.cpu().numpy())
                dice_score += (1. - loss).item()
                # pred, tempy = diceProprocess(pred, tempy)
                # loss = dice_loss(pred, tempy)
                batch_dice_list.append(loss.cpu().numpy())

            batch_dice = np.asarray(batch_dice_list, np.float32).mean()

            t = time.strftime('%X %x %Z')
            print(t, 'n', n, 'loss', batch_dice, "DICE", dice_score / len(batch_dice_list), len(batch_dice_list))
            loss_list.append(batch_dice)
            np.savetxt(loss_file, np.asarray(loss_list))

    if ((n + 1) % config_train['snapshot_iteration'] == 0):
        torch.save({
            'iteration': n + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
        }, config_train['model_save_prefix'] + "_{0:}.pt".format(n + 1))
        print("Saving snapshot")



# In[ ]:




