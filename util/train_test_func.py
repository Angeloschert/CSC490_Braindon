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
import torch
import numpy as np
from util.data_process import *



def volume_probability_prediction(temp_imgs, data_shape, label_shape, data_channel,
                                  class_num, batch_size, net):
    '''
    Test one image with sub regions along z-axis
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    [D, H, W] = temp_imgs[0].shape
    print(D, H, W)
    input_center = [int(D / 2), int(H / 2), int(W / 2)]
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_baches = []
    for center_slice in range(int(label_shape[0] / 2), D + int(label_shape[0] / 2), label_shape[0]):
        center_slice = min(center_slice, D - int(label_shape[0] / 2))
        sub_image_bach = []
        for chn in range(data_channel):
            temp_input_center = [center_slice, input_center[1], input_center[2]]
            # print(temp_input_center)
            # print(int(label_shape[0] / 2), D + int(label_shape[0] / 2), label_shape[0])
            # print(center_slice, D - int(label_shape[0] / 2))
            sub_image = extract_roi_from_volume(
                temp_imgs[chn], temp_input_center, data_shape)
            sub_image_bach.append(sub_image)
        sub_image_bach = np.asanyarray(sub_image_bach, np.float32)
        sub_image_baches.append(sub_image_bach)
    total_batch = len(sub_image_baches)
    max_mini_batch = int((total_batch + batch_size - 1) / batch_size)
    sub_label_idx = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_baches[mini_batch_idx * batch_size:
                                           min((mini_batch_idx + 1) * batch_size, total_batch)]
        if (mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx * batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size=[data_channel] + data_shape))
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        #print("before", data_mini_batch.shape)
        data_mini_batch = np.transpose(data_mini_batch, [0, 1, 3, 4, 2])
        # prob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch})
        data_mini_batch = torch.from_numpy(data_mini_batch).to(device)
        #print("after", data_mini_batch.shape)
        with torch.no_grad():
            predicty = net(data_mini_batch)
        #print('predicty',predicty.shape)
        softmax = torch.nn.Softmax(dim=1)
        prob_mini_batch = softmax(predicty)
        # prob_mini_batch = proby(data_mini_batch)
        prob_mini_batch = prob_mini_batch.cpu().numpy()
        prob_mini_batch =  np.transpose(prob_mini_batch, [0, 4, 2, 3, 1])
        #print("prob_mini_batch", prob_mini_batch.shape)
        for batch_idx in range(prob_mini_batch.shape[0]):
            center_slice = sub_label_idx * label_shape[0] + int(label_shape[0] / 2)
            center_slice = min(center_slice, D - int(label_shape[0] / 2))
            temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num / 2)]
            # print(prob_mini_batch[batch_idx].shape, prob_mini_batch.shape)
            sub_prob = np.reshape(prob_mini_batch[batch_idx], label_shape + [class_num])
            temp_prob = set_roi_to_volume(temp_prob, temp_input_center, sub_prob)
            sub_label_idx = sub_label_idx + 1
    return temp_prob



def volume_probability_prediction_3d_roi(temp_imgs, data_shape, label_shape, data_channel,
                                         class_num, batch_size, proby):
    '''
    Test one image with sub regions along x, y, z axis
    '''
    [D, H, W] = temp_imgs[0].shape
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_batches = []
    sub_image_centers = []
    roid_half = int(label_shape[0] / 2)
    roih_half = int(label_shape[1] / 2)
    roiw_half = int(label_shape[2] / 2)
    for centerd in range(roid_half, D + roid_half, label_shape[0]):
        centerd = min(centerd, D - roid_half)
        for centerh in range(roih_half, H + roih_half, label_shape[1]):
            centerh = min(centerh, H - roih_half)
            for centerw in range(roiw_half, W + roiw_half, label_shape[2]):
                centerw = min(centerw, W - roiw_half)
                temp_input_center = [centerd, centerh, centerw]
                sub_image_centers.append(temp_input_center)
                sub_image_batch = []
                for chn in range(data_channel):
                    sub_image = extract_roi_from_volume(temp_imgs[chn], temp_input_center, data_shape)
                    sub_image_batch.append(sub_image)
                sub_image_bach = np.asanyarray(sub_image_batch, np.float32)
                sub_image_batches.append(sub_image_bach)

    total_batch = len(sub_image_batches)
    max_mini_batch = int((total_batch + batch_size - 1) / batch_size)
    sub_label_idx = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_batches[mini_batch_idx * batch_size:
                                            min((mini_batch_idx + 1) * batch_size, total_batch)]
        if (mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx * batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size=[data_channel] + data_shape))
        data_mini_batch = np.asanyarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        # outprob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch})
        predicty = net(data_mini_batch, is_training=True)
        softmax = torch.nn.Softmax(dim=-1)
        outprob_mini_batch = softmax(predicty)
        # outprob_mini_batch = proby(data_mini_batch)
        for batch_idx in range(batch_size):
            glb_batch_idx = batch_idx + mini_batch_idx * batch_size
            if (glb_batch_idx >= total_batch):
                continue
            temp_center = sub_image_centers[glb_batch_idx]
            temp_prob = set_roi_to_volume(temp_prob, temp_center + [1], outprob_mini_batch[batch_idx])
            sub_label_idx = sub_label_idx + 1
    return temp_prob


def volume_probability_prediction_dynamic_shape(temp_imgs, data_shape, label_shape, data_channel,
                                                class_num, batch_size, net):
    '''
    Test one image with sub regions along z-axis
    The height and width of input tensor is adapted to those of the input image
    '''
    # construct graph
    [D, H, W] = temp_imgs[0].shape
    Hx = max(int((H + 3) / 4) * 4, data_shape[1])
    Wx = max(int((W + 3) / 4) * 4, data_shape[2])
    data_slice = data_shape[0]
    label_slice = label_shape[0]
    full_data_shape = [batch_size, data_slice, Hx, Wx, data_channel]
    
    # x = tf.placeholder(torch.float32, full_data_shape)
    # x = torch.zeros(shape=full_data_shape, dtype=torch.float32, requires_grad=True)
    #predicty = net(x, is_training=True)
    #softmax = torch.nn.Softmax(dim=-1)
    # proby = softmax(predicty)

    new_data_shape = [data_slice, Hx, Wx]
    new_label_shape = [label_slice, Hx, Wx]
    temp_prob = volume_probability_prediction(temp_imgs, new_data_shape, new_label_shape, data_channel,
                                              class_num, batch_size, net)
    return temp_prob


def test_one_image_three_nets_adaptive_shape(temp_imgs, data_shapes, label_shapes, data_channel, class_num,
                                             batch_size, nets, shape_mode):
    '''
    Test one image with three anisotropic networks with fixed or adaptable tensor height and width.
    These networks are used in axial, saggital and coronal view respectively.
    shape_mode: 0: use fixed tensor shape in all direction
                1: compare tensor shape and image shape and then select fixed or adaptive tensor shape
                2: use adaptive tensor shape in all direction
    '''
    print('======================')
    [ax_data_shape, sg_data_shape, cr_data_shape] = data_shapes
    [ax_label_shape, sg_label_shape, cr_label_shape] = label_shapes
    [D, H, W] = temp_imgs[0].shape
    print('======================')
    print(D, H, W)
    if (shape_mode == 0 or (shape_mode == 1 and (H <= ax_data_shape[1] and W <= ax_data_shape[2]))):
        prob = volume_probability_prediction(temp_imgs, ax_data_shape, ax_label_shape, data_channel,
                                             class_num, batch_size, nets[0])
    else:
        prob = volume_probability_prediction_dynamic_shape(temp_imgs, ax_data_shape, ax_label_shape, data_channel,
                                                           class_num, batch_size, nets[0])

    tr_volumes1 = transpose_volumes(temp_imgs, 'sagittal')
    [sgD, sgH, sgW] = tr_volumes1[0].shape
    if (shape_mode == 0 or (shape_mode == 1 and (sgH <= sg_data_shape[1] and sgW <= sg_data_shape[2]))):
        prob1 = volume_probability_prediction(tr_volumes1, sg_data_shape, sg_label_shape, data_channel,
                                              class_num, batch_size, nets[1])
    else:
        prob1 = volume_probability_prediction_dynamic_shape(tr_volumes1, sg_data_shape, sg_label_shape, data_channel,
                                                            class_num, batch_size, nets[1])
    prob1 = np.transpose(prob1, [1, 2, 0, 3])

    tr_volumes2 = transpose_volumes(temp_imgs, 'coronal')
    [trD, trH, trW] = tr_volumes2[0].shape
    if (shape_mode == 0 or (shape_mode == 1 and (trH <= cr_data_shape[1] and trW <= cr_data_shape[2]))):
        prob2 = volume_probability_prediction(tr_volumes2, cr_data_shape, cr_label_shape, data_channel,
                                              class_num, batch_size, nets[2])
    else:
        prob2 = volume_probability_prediction_dynamic_shape(tr_volumes2, cr_data_shape, cr_label_shape, data_channel,
                                                            class_num, batch_size, nets[2])
    prob2 = np.transpose(prob2, [1, 0, 2, 3])

    prob = (prob + prob1 + prob2) / 3.0
    return prob
