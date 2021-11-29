from __future__ import absolute_import, print_function
from util.train_test_func import *
import torch
import torch.nn as nn
from crfseg import CRF
import numpy as np
from scipy import ndimage
import time
import os
import sys
import torch
from util.data_loader import *
from util.data_process import *
from util.parse_config import parse_config
from util.train_test_func import *
from util.MSNet import MSNet
from util.data_process import load_3d_volume_as_array, binary_dice3d


def test(config_file, crf=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = load_configs(config_file)
    config_test = configs["config_test"]
    batch_size = configs["batch_size"]

    net1, net1ax, net1sg, net1cr, class_num1, class_num1_ax, class_num1_sg, class_num1_cr, data_shape1, data_shape1_ax, data_shape1_sg, data_shape1_cr, label_shape1, label_shape1_ax, label_shape1_sg, label_shape1_cr = load_network(device, configs, "config_net1", 'network1ax', 'network1sg', 'network1cr', batch_size)
    net2, net2ax, net2sg, net2cr, class_num2, class_num2_ax, class_num2_sg, class_num2_cr, data_shape2, data_shape2_ax, data_shape2_sg, data_shape2_cr, label_shape2, label_shape2_ax, label_shape2_sg, label_shape2_cr = None, None, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None
    net3, net3ax, net3sg, net3cr, class_num3, class_num3_ax, class_num3_sg, class_num3_cr, data_shape3, data_shape3_ax, data_shape3_sg, data_shape3_cr, label_shape3, label_shape3_ax, label_shape3_sg, label_shape3_cr = None, None, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None

    if (config_test.get('whole_tumor_only', False) is False):
        net2, net2ax, net2sg, net2cr, class_num2, class_num2_ax, class_num2_sg, class_num2_cr, data_shape2, data_shape2_ax, data_shape2_sg, data_shape2_cr, label_shape2, label_shape2_ax, label_shape2_sg, label_shape2_cr = load_network(device, configs, "config_net2", 'network2ax', 'network2sg', 'network2cr', batch_size)
        net3, net3ax, net3sg, net3cr, class_num3, class_num3_ax, class_num3_sg, class_num3_cr, data_shape3, data_shape3_ax, data_shape3_sg, data_shape3_cr, label_shape3, label_shape3_ax, label_shape3_sg, label_shape3_cr = load_network(device, configs, "config_net3", 'network3ax', 'network3sg', 'network3cr', batch_size)
    
    print(os.curdir)
    config_data = configs["config_data"]
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    image_num = dataloader.get_total_image_number()

    # 3, start to test
    test_slice_direction = config_test.get('test_slice_direction', 'all')
    save_folder = config_data['save_folder']
    test_time = []
    struct = ndimage.generate_binary_structure(3, 2)
    margin = config_test.get('roi_patch_margin', 5)
    print("image_num",image_num)

    final_labels = []
    for i in range(image_num):
        [temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)

        print("temp imgs: " + str(np.array(temp_imgs).shape))
        print("temp weight: " + str(temp_weight.shape))
        print("temp name: " + str(temp_name))
        print("image names: " + str(img_names))
        print("temp bbox: " + str(temp_bbox))
        print("temp size: " + str(temp_size))

        t0 = time.time()
        pred1 = test1(device, crf, batch_size, temp_imgs, temp_weight, temp_name, img_names,
                     temp_bbox, temp_size, net1, net1ax, net1sg, net1cr,
                     class_num1, class_num1_ax, class_num1_sg, class_num1_cr,
                     data_shape1, data_shape1_ax, data_shape1_sg, data_shape1_cr, 
                     label_shape1, label_shape1_ax, label_shape1_sg, label_shape1_cr, 1)

        wt_threshold = 2000
        if (config_test.get('whole_tumor_only', False) is True):
            pred1_lc = ndimage.morphology.binary_closing(pred1, structure=struct)
            print("pred1_lc right after morphology.binary_closing: " + str(pred1_lc.shape))

            pred1_lc = get_largest_two_component(pred1_lc, False, wt_threshold)
            print("pred1_lc right after get_largest_two_component: " + str(pred1_lc.shape))

            out_label = pred1_lc
            print("out_label: " + str(out_label.shape))
        else:
            print("Not whole tumor")
            pred2, bbox1 = test2(device, crf, pred1, struct, batch_size, temp_imgs,
                                 temp_weight, temp_name, img_names, temp_bbox,
                                 temp_size, net2, net2ax, net2sg, net2cr,
                                 class_num2, class_num2_ax, class_num2_sg,
                                 class_num2_cr, data_shape2, data_shape2_ax,
                                 data_shape2_sg, data_shape2_cr, label_shape2,
                                 label_shape2_ax, label_shape2_sg,
                                 label_shape2_cr, 2)
            
            pred2, bbox2 = test3(device, crf, pred2, struct, batch_size, temp_imgs,
                                 temp_weight, temp_name, img_names, temp_bbox,
                                 temp_size, net3, net3ax, net3sg, net3cr,
                                 class_num3, class_num3_ax, class_num3_sg,
                                 class_num3_cr, data_shape3, data_shape3_ax,
                                 data_shape3_sg, data_shape3_cr, label_shape3,
                                 label_shape3_ax, label_shape3_sg,
                                 label_shape3_cr, 3)
            
            out_label = fuse(pred1, pred2, pred3, bbox1, bbox2)
            print("out_label: " + str(out_label.shape))
        
        # print(pred.sum())
        test_time.append(time.time() - t0)
        final_label = np.zeros(temp_size, np.int16)
        final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)
        final_labels.append((temp_name, final_label))
        print("final_label: " + str(final_label.shape))
        print("\n")

        save_array_as_nifty_volume(final_label, save_folder + "/{0:}.nii.gz".format(temp_name), img_names[0])

    return final_labels

# Model generating factory

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()

class CRFactory(object):
    @staticmethod
    def create(name):
        if name == 'gaussian_crf':
            crf_model = nn.Sequential(
                nn.Identity(),
                CRF(n_spatial_dims=3)
            )
            return crf_model
        # add your own networks here
        print('unsupported network:', name)
        exit()

# utils
def expand_n_channels(batch_size, n_channel, final_label):
    spatial = tuple(final_label.shape)
    result = np.zeros((batch_size, n_channel) + final_label.shape)
    for i in range(batch_size):
        for j in range(n_channel):
            result[i, j, :, :] = final_label
    return result

config_file = 'CSC490_Braindon/config17/test_wt.txt'

# 1, load configure file
def load_configs(config_file):
    config = parse_config(config_file)
    config_data = config.get('data')
    config_net1 = config.get('network1', None)
    config_net2 = config.get('network2', None)
    config_net3 = config.get('network3', None)
    config_test = config.get('testing')
    batch_size = config_test.get('batch_size', 5)

    configs = {
        "config": config,
        "config_data": config_data,
        "config_net1": config_net1,
        "config_net2": config_net2,
        "config_net3": config_net3,
        "config_test": config_test,
        "batch_size": batch_size
    }

    return configs

# 2, network for whole tumor, tumor core and enhanced tumor core
def load_network(device, configs, net_name, net_ax_name, net_sg_name, net_cr_name, batch_size):
    config = configs["config"]
    config_net = configs[net_name]

    net = net_ax = net_sg = net_cr = None
    class_num = class_num_ax = class_num_sg = class_num_cr = 0
    data_shape = data_shape_ax = data_shape_sg = data_shape_cr = None
    label_shape = label_shape_ax = label_shape_sg = label_shape_cr = None
    
    if (config_net):
        print("Has net{}".format(net_name[-1]))
        net_type = config_net['net_type']
        net_name = config_net['net_name']
        data_shape = config_net['data_shape']
        label_shape = config_net['label_shape']
        class_num = config_net['class_num']
        model_save_prefix = "CSC490_Braindon/" + config_net['model_save_prefix'] + ".pt"

        # construct graph for network
        full_data_shape = [batch_size] + data_shape
        
        net_class = NetFactory.create(net_type)
        net = net_class(num_classes=class_num, w_reg=None,
                        b_reg=None, in_chns=full_data_shape[-1])
        net.load_state_dict(torch.load(model_save_prefix, map_location=device)["model_state_dict"])
    else:
        print("No net{}".format(net_name[-1]))
        config_net_ax = config[net_ax_name]
        config_net_sg = config[net_sg_name]
        config_net_cr = config[net_cr_name]

        # construct graph for network axial
        net_type_ax = config_net_ax['net_type']
        net_name_ax = config_net_ax['net_name']
        data_shape_ax = config_net_ax['data_shape']
        label_shape_ax = config_net_ax['label_shape']
        class_num_ax = config_net_ax['class_num']
        model_save_prefix_ax = config_net_ax['model_save_prefix'] + ".pt"

        full_data_shape_ax = [batch_size] + data_shape_ax
        net_class_ax = NetFactory.create(net_type_ax)
        net_ax = net_class_ax(num_classes=class_num_ax, w_reg=None,
                            b_reg=None, in_chns=full_data_shape_ax[-1])
        net_ax.load_state_dict(torch.load(model_save_prefix_ax, map_location=device)["model_state_dict"])

        # construct graph for network sagittal
        net_type_sg = config_net_sg['net_type']
        net_name_sg = config_net_sg['net_name']
        data_shape_sg = config_net_sg['data_shape']
        label_shape_sg = config_net_sg['label_shape']
        class_num_sg = config_net_sg['class_num']
        model_save_prefix_sg = config_net_sg['model_save_prefix'] + ".pt"

        full_data_shape_sg = [batch_size] + data_shape_sg
        net_class_sg = NetFactory.create(net_type_sg)
        net_sg = net_class_sg(num_classes=class_num_sg, w_reg=None,
                            b_reg=None, in_chns=full_data_shape_sg[-1])
        net_sg.load_state_dict(torch.load(model_save_prefix_sg, map_location=device)["model_state_dict"])

        # construct graph for network corogal
        net_type_cr = config_net_cr['net_type']
        net_name_cr = config_net_cr['net_name']
        data_shape_cr = config_net_cr['data_shape']
        label_shape_cr = config_net_cr['label_shape']
        class_num_cr = config_net_cr['class_num']
        model_save_prefix_cr = config_net_cr['model_save_prefix'] + ".pt"

        full_data_shape_cr = [batch_size] + data_shape_cr
        net_class_cr = NetFactory.create(net_type_cr)
        net_cr = net_class_cr(num_classes=class_num_cr, w_reg=None,
                            b_reg=None, in_chns=full_data_shape_cr[-1])
        net_cr.load_state_dict(torch.load(model_save_prefix_cr, map_location=device)["model_state_dict"])
    return [net, net_ax, net_sg, net_cr, class_num, class_num_ax, class_num_sg,
            class_num_cr, data_shape, data_shape_ax, data_shape_sg,
            data_shape_cr, label_shape, label_shape_ax,
            label_shape_sg, label_shape_cr]

def test1(device, crf, batch_size, temp_imgs, temp_weight, temp_name, img_names, temp_bbox,
          temp_size, net, net_ax, net_sg, net_cr, class_num, class_num_ax,
          class_num_sg, class_num_cr, data_shape, data_shape_ax, data_shape_sg,
          data_shape_cr, label_shape, label_shape_ax, label_shape_sg,
          label_shape_cr, netid):
# ================================== test of 1st network ==================================
    if (net):
        data_shapes = [data_shape[:-1], data_shape[:-1], data_shape[:-1]]
        label_shapes = [label_shape[:-1], label_shape[:-1], label_shape[:-1]]
        nets = [net, net, net]
        #inputs = [x1, x1, x1]
    else:
        data_shapes = [data_shape_ax[:-1], data_shape_sg[:-1], data_shape_cr[:-1]]
        data_shape = data_shape_ax[-1]
        label_shapes = [label_shape_ax[:-1], label_shape_sg[:-1], label_shape_cr[:-1]]
        nets = [net_ax, net_sg, net_cr]
        #inputs = [x1ax, x1sg, x1cr]
        class_num = class_num_ax
    for i in range(len(nets)):
        nets[i] = nets[i].to(device)
    # print('=' * 20, "Going to prediction")
    prob = test_one_image_three_nets_adaptive_shape(temp_imgs, data_shapes, label_shapes, data_shape_ax[-1],
                                                     class_num, batch_size, nets, shape_mode=2)
    print("prob{} size: ".format(netid) + str(prob.shape))

    # ================== CRF ==================================
    if crf:
        crf_model = CRFactory.create("gaussian_crf")
        d, h, w, c = prob.shape
        prob = prob.reshape(c, d, h, w)
        prob = np.expand_dims(prob, axis=0)
        prob = crf_model(torch.tensor(prob, dtype=torch.float32))
        print("prob size after crf: ".format(netid) + str(prob.shape))

        prob = prob.detach().numpy()[0].reshape(d, h, w, c)
    # ===========================================================
    pred = np.asarray(np.argmax(prob, axis=3), np.uint16)
    pred = pred * temp_weight
    print("pred{} size".format(netid) + str(pred.shape))
    # ================================== End of 1st network ==================================
    return pred


def test2(device, crf, pred, struct, batch_size, temp_imgs, temp_weight, temp_name,
           img_names, temp_bbox, temp_size, net, net_ax, net_sg, net_cr,
           class_num, class_num_ax, class_num_sg, class_num_cr, data_shape,
           data_shape_ax, data_shape_sg, data_shape_cr, label_shape,
           label_shape_ax, label_shape_sg, label_shape_cr, netid):
    # ================================== test of 2nd network ==================================
    if (pred.sum() == 0):
        print('net{} output is null'.format(netid), temp_name)
        bbox = get_ND_bounding_box(temp_imgs[0] > 0, margin)
    else:
        pred_lc = ndimage.morphology.binary_closing(pred, structure=struct)
        pred_lc = get_largest_two_component(pred_lc, False, wt_threshold)
        bbox = get_ND_bounding_box(pred_lc, margin)
    sub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox[0], bbox[1]) for one_img in temp_imgs]
    sub_weight = crop_ND_volume_with_bounding_box(temp_weight, bbox[0], bbox[1])

    if (net):
        data_shapes = [data_shape[:-1], data_shape[:-1], data_shape[:-1]]
        label_shapes = [label_shape[:-1], label_shape[:-1], label_shape[:-1]]
        nets = [net, net, net]
        #inputs = [x2, x2, x2]
    else:
        data_shapes = [data_shape_ax[:-1], data_shape_sg[:-1], data_shape_cr[:-1]]
        label_shapes = [label_shape_ax[:-1], label_shape_sg[:-1], label_shape_cr[:-1]]
        nets = [net_ax, net_sg, net_cr]
        #inputs = [x2ax, x2sg, x2cr]
        class_num = class_num_ax
    for i in range(len(nets)):
        nets[i] = nets[i].to(device)
    prob = test_one_image_three_nets_adaptive_shape(sub_imgs, data_shapes, label_shapes, data_shape_ax[-1],
                                                        class_num, batch_size, nets, shape_mode=1)
    print("prob{} size: ".format(netid) + str(prob.shape).format(netid))
    
    # ================== CRF ==================================
    if crf:
        crf_model = CRFactory.create("gaussian_crf")
        d, h, w, c = prob.shape
        prob = prob.reshape(c, d, h, w)
        prob = np.expand_dims(prob, axis=0)
        prob = crf_model(torch.tensor(prob, dtype=torch.float32))
        print("prob{} size after crf: ".format(netid) + str(prob.shape))

        prob = prob.detach().numpy()[0].reshape(d, h, w, c)
    # ===========================================================
    pred = np.asarray(np.argmax(prob, axis=3), np.uint16)
    pred = pred * sub_weight
    print("pred{} size: ".format(netid) + str(pred.shape))
    # ================================== End of 2nd network ==================================
    return pred, bbox


def test3(device, crf, pred, struct, batch_size, temp_imgs, temp_weight, temp_name,
           img_names, temp_bbox, temp_size, net, net_ax, net_sg, net_cr,
           class_num, class_num_ax, class_num_sg, class_num_cr, data_shape,
           data_shape_ax, data_shape_sg, data_shape_cr, label_shape,
           label_shape_ax, label_shape_sg, label_shape_cr, netid):
    # ================================== test of 3rd network ==================================
    if (pred2.sum() == 0):
        [roid, roih, roiw] = sub_imgs[0].shape
        bbox = [[0, 0, 0], [roid - 1, roih - 1, roiw - 1]]
        subsub_imgs = sub_imgs
        subsub_weight = sub_weight
    else:
        pred_lc = ndimage.morphology.binary_closing(pred, structure=struct)
        pred_lc = get_largest_two_component(pred_lc)
        bbox = get_ND_bounding_box(pred_lc, margin)
        subsub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox[0], bbox[1]) for one_img in sub_imgs]
        subsub_weight = crop_ND_volume_with_bounding_box(sub_weight, bbox[0], bbox[1])

    if (net):
        data_shapes = [data_shape[:-1], data_shape[:-1], data_shape[:-1]]
        label_shapes = [label_shape[:-1], label_shape[:-1], label_shape[:-1]]
        nets = [net, net, net]
        #inputs = [x3, x3, x3]
    else:
        data_shapes = [data_shape_ax[:-1], data_shape_sg[:-1], data_shape_cr[:-1]]
        label_shapes = [label_shape_ax[:-1], label_shape_sg[:-1], label_shape_cr[:-1]]
        nets = [net_ax, net_sg, net_cr]
        #inputs = [x3ax, x3sg, x3cr]
        class_num = class_num_ax
    for i in range(len(nets)):
        nets[i] = nets[i].to(device)
    prob = test_one_image_three_nets_adaptive_shape(sub_imgs, data_shapes, label_shapes, data_shape_ax[-1],
                                                        class_num, batch_size, nets, shape_mode=1)
    print("prob{} size: ".format(netid) + str(prob.shape).format(netid))
    
    # ================== CRF ==================================
    if crf:
        crf_model = CRFactory.create("gaussian_crf")
        d, h, w, c = prob.shape
        prob = prob.reshape(c, d, h, w)
        prob = np.expand_dims(prob, axis=0)
        prob = crf_model(torch.tensor(prob, dtype=torch.float32))
        print("prob{} size after crf: ".format(netid) + str(prob.shape))

        prob = prob.detach().numpy()[0].reshape(d, h, w, c)
    # ===========================================================
    pred = np.asarray(np.argmax(prob, axis=3), np.uint16)
    pred = pred * sub_weight
    print("pred{} size: ".format(netid) + str(pred.shape))
    # ================================== End of 3rd network ==================================
    return pred, bbox


def fuse(pred1, pred2, pred3, bbox1, bbox2):
    # 5.4, fuse results at 3 levels
    # convert subsub_label to full size (non-enhanced)
    label3_roi = np.zeros_like(pred2)
    label3_roi = set_ND_volume_roi_with_bounding_box_range(label3_roi, bbox2[0], bbox2[1], pred3)
    label3 = np.zeros_like(pred1)
    label3 = set_ND_volume_roi_with_bounding_box_range(label3, bbox1[0], bbox1[1], label3_roi)

    label2 = np.zeros_like(pred1)
    label2 = set_ND_volume_roi_with_bounding_box_range(label2, bbox1[0], bbox1[1], pred2)

    label1_mask = (pred1 + label2 + label3) > 0
    label1_mask = ndimage.morphology.binary_closing(label1_mask, structure=struct)
    label1_mask = get_largest_two_component(label1_mask, False, wt_threshold)
    label1 = pred1 * label1_mask

    label2_3_mask = (label2 + label3) > 0
    label2_3_mask = label2_3_mask * label1_mask
    label2_3_mask = ndimage.morphology.binary_closing(label2_3_mask, structure=struct)
    label2_3_mask = remove_external_core(label1, label2_3_mask)

    if (label2_3_mask.sum() > 0):
        label2_3_mask = get_largest_two_component(label2_3_mask)

    label1 = (label1 + label2_3_mask) > 0
    label2 = label2_3_mask
    label3 = label2 * label3
    vox_3 = np.asarray(label3 > 0, np.float32).sum()

    if (0 < vox_3 and vox_3 < 30):
        label3 = np.zeros_like(label2)

    # 5.5, convert label and save output
    out_label = label1 * 2
    if ('Flair' in config_data['modality_postfix'] and 'mha' in config_data['file_postfix']):
        out_label[label2 > 0] = 3
        out_label[label3 == 1] = 1
        out_label[label3 == 2] = 4
    elif ('flair' in config_data['modality_postfix'] and 'nii' in config_data['file_postfix']):
        out_label[label2 > 0] = 1
        out_label[label3 > 0] = 4
    out_label = np.asarray(out_label, np.int16)
    
    return out_label


if __name__ == '__main__':
    if (len(sys.argv) ==2):
        print("about to segment images without CRF")
        config_file = str(sys.argv[1])
        assert (os.path.isfile(config_file))
        test(config_file,False)
    elif len(sys.argv)==3 and sys.argv[2]=="crf":
        print("about to segment images with CRF")
        config_file = str(sys.argv[1])
        assert (os.path.isfile(config_file))
        test(config_file,True)