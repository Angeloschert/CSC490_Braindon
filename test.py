from util.train_test_func import *
import torch
import torch.nn as nn
from crfseg import CRF

crf_model = nn.Sequential(
    nn.Identity(),
    CRF(n_spatial_dims=3)
)

from __future__ import absolute_import, print_function
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

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(len(sys.argv) <= 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/test_all_class.txt')
        exit()
config_file = str(sys.argv[1])
assert(os.path.isfile(config_file))

WITH_CRF = False
if sys.argv[2] == "crf":
    WITH_CRF = True

print("Will testing with crf.")







# 1, load configure file
config = parse_config(config_file)
config_data = config.get('data')
config_net1 = config.get('network1', None)
config_net2 = config.get('network2', None)
config_net3 = config.get('network3', None)
config_test = config.get('testing')
batch_size = config_test.get('batch_size', 5)

# 2.1, network for whole tumor
if (config_net1):
    print("Has net1")
    net_type1 = config_net1['net_type']
    net_name1 = config_net1['net_name']
    data_shape1 = config_net1['data_shape']
    label_shape1 = config_net1['label_shape']
    class_num1 = config_net1['class_num']
    model_save_prefix1 = config_net1['model_save_prefix'] + ".pt"

    # construct graph for 1st network
    full_data_shape1 = [batch_size] + data_shape1
    
    net_class1 = NetFactory.create(net_type1)
    net1 = net_class1(num_classes=class_num1, w_reg=None,
                      b_reg=None, in_chns=full_data_shape1[-1])
    net1.load_state_dict(torch.load(model_save_prefix1, map_location=device)["model_state_dict"])
else:
    print("No net 1")
    config_net1ax = config['network1ax']
    config_net1sg = config['network1sg']
    config_net1cr = config['network1cr']

    # construct graph for 1st network axial
    net_type1ax = config_net1ax['net_type']
    net_name1ax = config_net1ax['net_name']
    data_shape1ax = config_net1ax['data_shape']
    label_shape1ax = config_net1ax['label_shape']
    class_num1ax = config_net1ax['class_num']
    model_save_prefix1ax = config_net1ax['model_save_prefix'] + ".pt"

    full_data_shape1ax = [batch_size] + data_shape1ax
    net_class1ax = NetFactory.create(net_type1ax)
    net1ax = net_class1ax(num_classes=class_num1ax, w_reg=None,
                          b_reg=None, in_chns=full_data_shape1ax[-1])
    net1ax.load_state_dict(torch.load(model_save_prefix1ax, map_location=device)["model_state_dict"])

    # construct graph for 1st network sagittal
    net_type1sg = config_net1sg['net_type']
    net_name1sg = config_net1sg['net_name']
    data_shape1sg = config_net1sg['data_shape']
    label_shape1sg = config_net1sg['label_shape']
    class_num1sg = config_net1sg['class_num']
    model_save_prefix1sg = config_net1sg['model_save_prefix'] + ".pt"

    full_data_shape1sg = [batch_size] + data_shape1sg
    net_class1sg = NetFactory.create(net_type1sg)
    net1sg = net_class1sg(num_classes=class_num1sg, w_reg=None,
                          b_reg=None, in_chns=full_data_shape1sg[-1])
    net1sg.load_state_dict(torch.load(model_save_prefix1sg, map_location=device)["model_state_dict"])

    # construct graph for 1st network corogal
    net_type1cr = config_net1cr['net_type']
    net_name1cr = config_net1cr['net_name']
    data_shape1cr = config_net1cr['data_shape']
    label_shape1cr = config_net1cr['label_shape']
    class_num1cr = config_net1cr['class_num']
    model_save_prefix1cr = config_net1cr['model_save_prefix'] + ".pt"

    full_data_shape1cr = [batch_size] + data_shape1cr
    net_class1cr = NetFactory.create(net_type1cr)
    net1cr = net_class1cr(num_classes=class_num1cr, w_reg=None,
                          b_reg=None, in_chns=full_data_shape1cr[-1])
    net1cr.load_state_dict(torch.load(model_save_prefix1cr, map_location=device)["model_state_dict"])

if (config_test.get('whole_tumor_only', False) is False):
    # 2.2, networks for tumor core
    if (config_net2):
        net_type2 = config_net2['net_type']
        net_name2 = config_net2['net_name']
        data_shape2 = config_net2['data_shape']
        label_shape2 = config_net2['label_shape']
        class_num2 = config_net2['class_num']
        model_save_prefix2 = config_net2['model_save_prefix'] + ".pt"

        # construct graph for 2st network
        full_data_shape2 = [batch_size] + data_shape2
        net_class2 = NetFactory.create(net_type2)
        net2 = net_class2(num_classes=class_num2, w_reg=None,
                          b_reg=None, in_chns=full_data_shape2[-1])
        net2.load_state_dict(torch.load(model_save_prefix2, map_location=device)["model_state_dict"])
    else:
        config_net2ax = config['network2ax']
        config_net2sg = config['network2sg']
        config_net2cr = config['network2cr']

        # construct graph for 2st network axial
        net_type2ax = config_net2ax['net_type']
        net_name2ax = config_net2ax['net_name']
        data_shape2ax = config_net2ax['data_shape']
        label_shape2ax = config_net2ax['label_shape']
        class_num2ax = config_net2ax['class_num']
        model_save_prefix2ax = config_net2ax['model_save_prefix'] + ".pt"

        full_data_shape2ax = [batch_size] + data_shape2ax

        net_class2ax = NetFactory.create(net_type2ax)
        net2ax = net_class2ax(num_classes=class_num2ax, w_reg=None,
                              b_reg=None, in_chns=full_data_shape2ax[-1])
        net2ax.load_state_dict(torch.load(model_save_prefix2ax, map_location=device)["model_state_dict"])

        # construct graph for 2st network sagittal
        net_type2sg = config_net2sg['net_type']
        net_name2sg = config_net2sg['net_name']
        data_shape2sg = config_net2sg['data_shape']
        label_shape2sg = config_net2sg['label_shape']
        class_num2sg = config_net2sg['class_num']
        model_save_prefix2sg = config_net2sg['model_save_prefix'] + ".pt"

        full_data_shape2sg = [batch_size] + data_shape2sg
        net_class2sg = NetFactory.create(net_type2sg)
        net2sg = net_class2sg(num_classes=class_num2sg, w_reg=None,
                              b_reg=None, in_chns=full_data_shape2sg[-1])
        net2sg.load_state_dict(torch.load(model_save_prefix2sg, map_location=device)["model_state_dict"])

        # construct graph for 2st network corogal
        net_type2cr = config_net2cr['net_type']
        net_name2cr = config_net2cr['net_name']
        data_shape2cr = config_net2cr['data_shape']
        label_shape2cr = config_net2cr['label_shape']
        class_num2cr = config_net2cr['class_num']
        model_save_prefix2cr = config_net2cr['model_save_prefix'] + ".pt"

        full_data_shape2cr = [batch_size] + data_shape2cr
        net_class2cr = NetFactory.create(net_type2cr)
        net2cr = net_class2cr(num_classes=class_num2cr, w_reg=None,
                              b_reg=None, in_chns=full_data_shape2cr[-1])
        net2cr.load_state_dict(torch.load(model_save_prefix2cr, map_location=device)["model_state_dict"])
        

    # 2.3, networks for enhanced tumor
    if (config_net3):
        net_type3 = config_net3['net_type']
        net_name3 = config_net3['net_name']
        data_shape3 = config_net3['data_shape']
        label_shape3 = config_net3['label_shape']
        class_num3 = config_net3['class_num']
        model_save_prefix3 = config_net1['model_save_prefix'] + ".pt"

        # construct graph for 3st network
        full_data_shape3 = [batch_size] + data_shape3
        net_class3 = NetFactory.create(net_type3)
        net3 = net_class3(num_classes=class_num3, w_reg=None,b_reg=None, in_chns=full_data_shape3[-1])
        net3.load_state_dict(torch.load(model_save_prefix3, map_location=device)["model_state_dict"])
    else:
        config_net3ax = config['network3ax']
        config_net3sg = config['network3sg']
        config_net3cr = config['network3cr']

        # construct graph for 3st network axial
        net_type3ax = config_net3ax['net_type']
        net_name3ax = config_net3ax['net_name']
        data_shape3ax = config_net3ax['data_shape']
        label_shape3ax = config_net3ax['label_shape']
        class_num3ax = config_net3ax['class_num']
        model_save_prefix3ax = config_net3ax['model_save_prefix'] + ".pt"

        full_data_shape3ax = [batch_size] + data_shape3ax
        net_class3ax = NetFactory.create(net_type3ax)
        net3ax = net_class3ax(num_classes=class_num3ax, w_reg=None,
                              b_reg=None, in_chns=full_data_shape3ax[-1])
        net3ax.load_state_dict(torch.load(model_save_prefix3ax, map_location=device)["model_state_dict"])

        # construct graph for 3st network sagittal
        net_type3sg = config_net3sg['net_type']
        net_name3sg = config_net3sg['net_name']
        data_shape3sg = config_net3sg['data_shape']
        label_shape3sg = config_net3sg['label_shape']
        class_num3sg = config_net3sg['class_num']
        model_save_prefix3sg = config_net3sg['model_save_prefix'] + ".pt"

        # construct graph for 3st network
        full_data_shape3sg = [batch_size] + data_shape3sg
        net_class3sg = NetFactory.create(net_type3sg)
        net3sg = net_class3sg(num_classes=class_num3sg, w_reg=None,
                              b_reg=None, in_chns=full_data_shape3sg[-1])
        net3sg.load_state_dict(torch.load(model_save_prefix3sg, map_location=device)["model_state_dict"])

        # construct graph for 3st network corogal
        net_type3cr = config_net3cr['net_type']
        net_name3cr = config_net3cr['net_name']
        data_shape3cr = config_net3cr['data_shape']
        label_shape3cr = config_net3cr['label_shape']
        class_num3cr = config_net3cr['class_num']
        model_save_prefix3cr = config_net3cr['model_save_prefix'] + ".pt"

        # construct graph for 3st network
        full_data_shape3cr = [batch_size] + data_shape3cr
        net_class3cr = NetFactory.create(net_type3cr)
        net3cr = net_class3cr(num_classes=class_num3cr, w_reg=None,
                              b_reg=None, in_chns=full_data_shape3cr[-1])
        net3cr.load_state_dict(torch.load(model_save_prefix3cr, map_location=device)["model_state_dict"])

import os
print(os.curdir)
dataloader = DataLoader(config_data)
dataloader.load_data()
image_num = dataloader.get_total_image_number()

# 5, start to test
test_slice_direction = config_test.get('test_slice_direction', 'all')
save_folder = config_data['save_folder']
test_time = []
struct = ndimage.generate_binary_structure(3, 2)
margin = config_test.get('roi_patch_margin', 5)
print("image_num",image_num)

final_labels = []
for i in range(image_num):
    print("processing {}/{}".format(i,image_num))

    [temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
    t0 = time.time()

# ================================== test of 1st network ==================================
    data_shape = None
    if (config_net1):
        data_shapes = [data_shape1[:-1], data_shape1[:-1], data_shape1[:-1]]
        data_shape = data_shape1
        label_shapes = [label_shape1[:-1], label_shape1[:-1], label_shape1[:-1]]
        nets = [net1, net1, net1]
        #inputs = [x1, x1, x1]
        class_num = class_num1
    else:
<<<<<<< HEAD
        data_shapes = [data_shape1ax[:-1], data_shape1sg[:-1], data_shape1cr[:-1]]
        data_shape = data_shape1ax[-1]
        label_shapes = [label_shape1ax[:-1], label_shape1sg[:-1], label_shape1cr[:-1]]
        nets = [net1ax, net1sg, net1cr]
        #inputs = [x1ax, x1sg, x1cr]
        class_num = class_num1ax
    for i in range(len(nets)):
        nets[i] = nets[i].to(device)
    # print('=' * 20, "Going to prediction")
    print("temp imgs: " + str(np.array(temp_imgs).shape))
    print("temp weight: " + str(temp_weight.shape))
    print("temp name: " + str(temp_name))
    print("image names: " + str(img_names))
    print("temp bbox: " + str(temp_bbox))
    print("temp size: " + str(temp_size))
    prob1 = test_one_image_three_nets_adaptive_shape(temp_imgs, data_shapes, label_shapes, data_shape1ax[-1],
                                                     class_num,
                                                     batch_size, nets, shape_mode=2)
    print("prob1 size: " + str(prob1.shape))

    # ================== CRF ==================================
    if WITH_CRF:
        d, h, w, c = prob1.shape
        prob1 = prob1.reshape(c, d, h, w)
        prob1 = np.expand_dims(prob1, axis=0)
        prob1 = crf_model(torch.tensor(prob1, dtype=torch.float32))
        print("prob1 size after crf: " + str(prob1.shape))

        prob1 = prob1.detach().numpy()[0].reshape(d, h, w, c)
    # ===========================================================
    pred1 = np.asarray(np.argmax(prob1, axis=3), np.uint16)
    pred1 = pred1 * temp_weight
    print("pred1 size" + str(pred1.shape))
    # ================================== End of 1st network ==================================



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



        # ================================== test of 2nd network ==================================
        if (pred1.sum() == 0):
            print('net1 output is null', temp_name)
            bbox1 = get_ND_bounding_box(temp_imgs[0] > 0, margin)
        else:
=======
        config_net1ax = config['network1ax']
        config_net1sg = config['network1sg']
        config_net1cr = config['network1cr']

        # construct graph for 1st network axial
        net_type1ax = config_net1ax['net_type']
        net_name1ax = config_net1ax['net_name']
        data_shape1ax = config_net1ax['data_shape']
        label_shape1ax = config_net1ax['label_shape']
        class_num1ax = config_net1ax['class_num']
        model_save_prefix1ax = config_net1ax['model_save_prefix'] + ".pt"

        full_data_shape1ax = [batch_size] + data_shape1ax
        # x1ax = tf.placeholder(tf.float32, shape = full_data_shape1ax)
        x1ax = torch.zeros(full_data_shape1ax, dtype=torch.float32, requires_grad=True)
        net_class1ax = NetFactory.create(net_type1ax)
        net1ax = net_class1ax(num_classes=class_num1ax, w_reg=None,
                              b_reg=None, in_chns=full_data_shape1ax[1])
        # net1ax.set_params(config_net1ax)
        net1ax.load_state_dict(torch.load(model_save_prefix1ax, map_location=device)["model_state_dict"])
        predicty1ax = net1ax(x1ax)
        softmax1ax = torch.nn.Softmax(dim=-1)
        # proby1ax = tf.nn.softmax(predicty1ax)
        proby1ax = softmax1ax(predicty1ax)

        # construct graph for 1st network sagittal
        net_type1sg = config_net1sg['net_type']
        net_name1sg = config_net1sg['net_name']
        data_shape1sg = config_net1sg['data_shape']
        label_shape1sg = config_net1sg['label_shape']
        class_num1sg = config_net1sg['class_num']
        model_save_prefix1sg = config_net1sg['model_save_prefix'] + ".pt"

        full_data_shape1sg = [batch_size] + data_shape1sg
        # x1sg = tf.placeholder(tf.float32, shape = full_data_shape1sg)
        x1sg = torch.zeros(full_data_shape1sg, dtype=torch.float32, requires_grad=True)
        net_class1sg = NetFactory.create(net_type1sg)
        net1sg = net_class1sg(num_classes=class_num1sg, w_reg=None,
                              b_reg=None, in_chns=full_data_shape1sg[1])
        # net1sg.set_params(config_net1sg)
        net1sg.load_state_dict(torch.load(model_save_prefix1sg, map_location=device)["model_state_dict"])
        predicty1sg = net1sg(x1sg)
        # proby1sg = tf.nn.softmax(predicty1sg)
        softmax1sg = torch.nn.Softmax(dim=-1)
        proby1sg = softmax1sg(predicty1sg)

        # construct graph for 1st network corogal
        net_type1cr = config_net1cr['net_type']
        net_name1cr = config_net1cr['net_name']
        data_shape1cr = config_net1cr['data_shape']
        label_shape1cr = config_net1cr['label_shape']
        class_num1cr = config_net1cr['class_num']
        model_save_prefix1cr = config_net1cr['model_save_prefix'] + ".pt"

        full_data_shape1cr = [batch_size] + data_shape1cr
        # x1cr = tf.placeholder(tf.float32, shape = full_data_shape1cr)
        x1cr = torch.zeros(full_data_shape1cr, dtype=torch.float32, requires_grad=True)
        net_class1cr = NetFactory.create(net_type1cr)
        net1cr = net_class1cr(num_classes=class_num1cr, w_reg=None,
                              b_reg=None, in_chns=full_data_shape1cr[1])
        # net1cr.set_params(config_net1cr)
        net1cr.load_state_dict(torch.load(model_save_prefix1cr, map_location=device)["model_state_dict"])
        predicty1cr = net1cr(x1cr)
        # proby1cr = tf.nn.softmax(predicty1cr)
        softmax1cr = torch.nn.Softmax(dim=-1)
        proby1cr = softmax1cr(predicty1cr)

    if (config_test.get('whole_tumor_only', False) is False):
        # 2.2, networks for tumor core
        if (config_net2):
            net_type2 = config_net2['net_type']
            net_name2 = config_net2['net_name']
            data_shape2 = config_net2['data_shape']
            label_shape2 = config_net2['label_shape']
            class_num2 = config_net2['class_num']
            model_save_prefix2 = config_net2['model_save_prefix'] + ".pt"

            # construct graph for 2st network
            full_data_shape2 = [batch_size] + data_shape2
            # x2 = tf.placeholder(tf.float32, shape = full_data_shape2)
            x2 = torch.zeros(full_data_shape2, dtype=torch.float32, requires_grad=True)
            net_class2 = NetFactory.create(net_type2)
            net2 = net_class2(num_classes=class_num2, w_reg=None,
                              b_reg=None, in_chns=full_data_shape2[1])
            # net2.set_params(config_net2)
            net2.load_state_dict(torch.load(model_save_prefix2, map_location=device)["model_state_dict"])
            predicty2 = net2(x2, is_training=True)
            # proby2 = tf.nn.softmax(predicty2)
            softmax2 = torch.nn.Softmax(dim=-1)
            proby2 = softmax2(predicty2)
        else:
            config_net2ax = config['network2ax']
            config_net2sg = config['network2sg']
            config_net2cr = config['network2cr']

            # construct graph for 2st network axial
            net_type2ax = config_net2ax['net_type']
            net_name2ax = config_net2ax['net_name']
            data_shape2ax = config_net2ax['data_shape']
            label_shape2ax = config_net2ax['label_shape']
            class_num2ax = config_net2ax['class_num']
            model_save_prefix2ax = config_net2ax['model_save_prefix'] + ".pt"

            full_data_shape2ax = [batch_size] + data_shape2ax
            # x2ax = tf.placeholder(tf.float32, shape = full_data_shape2ax)
            x2ax = torch.zeros(full_data_shape2ax, dtype=torch.float32, requires_grad=True)

            net_class2ax = NetFactory.create(net_type2ax)
            net2ax = net_class2ax(num_classes=class_num2ax, w_reg=None,
                                  b_reg=None, in_chns=full_data_shape2ax[1])
            # net2ax.set_params(config_net2ax)
            net2ax.load_state_dict(torch.load(model_save_prefix2ax, map_location=device)["model_state_dict"])
            predicty2ax = net2ax(x2ax, is_training=True)
            softmax2ax = torch.nn.Softmax(dim=-1)
            # proby2ax = tf.nn.softmax(predicty2ax)
            proby2ax = softmax2ax(predicty2ax)

            # construct graph for 2st network sagittal
            net_type2sg = config_net2sg['net_type']
            net_name2sg = config_net2sg['net_name']
            data_shape2sg = config_net2sg['data_shape']
            label_shape2sg = config_net2sg['label_shape']
            class_num2sg = config_net2sg['class_num']
            model_save_prefix2sg = config_net2sg['model_save_prefix'] + ".pt"

            full_data_shape2sg = [batch_size] + data_shape2sg
            # x2sg = tf.placeholder(tf.float32, shape = full_data_shape2sg)
            x2sg = torch.zeros(full_data_shape2sg, dtype=torch.float32, requires_grad=True)
            net_class2sg = NetFactory.create(net_type2sg)
            net2sg = net_class2sg(num_classes=class_num2sg, w_reg=None,
                                  b_reg=None, in_chns=full_data_shape2sg[1])
            # net2sg.set_params(config_net2sg)
            net2sg.load_state_dict(torch.load(model_save_prefix2sg, map_location=device)["model_state_dict"])
            predicty2sg = net2sg(x2sg)
            # proby2sg = tf.nn.softmax(predicty2sg)
            softmax2sg = torch.nn.Softmax(dim=-1)
            proby2sg = softmax2sg(predicty2sg)

            # construct graph for 2st network corogal
            net_type2cr = config_net2cr['net_type']
            net_name2cr = config_net2cr['net_name']
            data_shape2cr = config_net2cr['data_shape']
            label_shape2cr = config_net2cr['label_shape']
            class_num2cr = config_net2cr['class_num']
            model_save_prefix2cr = config_net2cr['model_save_prefix'] + ".pt"

            full_data_shape2cr = [batch_size] + data_shape2cr
            # x2cr = tf.placeholder(tf.float32, shape = full_data_shape2cr)
            x2cr = torch.zeros(full_data_shape2cr, dtype=torch.float32, requires_grad=True)
            net_class2cr = NetFactory.create(net_type2cr)
            net2cr = net_class2cr(num_classes=class_num2cr, w_reg=None,
                                  b_reg=None, in_chns=full_data_shape2cr[1])
            # net2cr.set_params(config_net2cr)
            net2cr.load_state_dict(torch.load(model_save_prefix2cr, map_location=device)["model_state_dict"])
            predicty2cr = net2cr(x2cr)
            # proby2cr = tf.nn.softmax(predicty2cr)
            softmax2cr = torch.nn.Softmax(dim=-1)
            proby2cr = softmax2cr(predicty2cr)

        # 2.3, networks for enhanced tumor
        if (config_net3):
            net_type3 = config_net3['net_type']
            net_name3 = config_net3['net_name']
            data_shape3 = config_net3['data_shape']
            label_shape3 = config_net3['label_shape']
            class_num3 = config_net3['class_num']
            model_save_prefix3 = config_net3['model_save_prefix'] + ".pt"

            # construct graph for 3st network
            full_data_shape3 = [batch_size] + data_shape3
            # x3 = tf.placeholder(tf.float32, shape = full_data_shape3)
            x3 = torch.zeros(full_data_shape3, dtype=torch.float32, requires_grad=True)
            net_class3 = NetFactory.create(net_type3)
            net3 = net_class3(num_classes=class_num3, w_reg=None,
                              b_reg=None, in_chns=full_data_shape3[1])
            # net3.set_params(config_net3)
            net3.load_state_dict(torch.load(model_save_prefix3, map_location=device)["model_state_dict"])
            predicty3 = net3(x3)
            # proby3 = tf.nn.softmax(predicty3)
            softmax3 = torch.nn.Softmax(dim=-1)
            proby3 = softmax3(predicty3)
        else:
            config_net3ax = config['network3ax']
            config_net3sg = config['network3sg']
            config_net3cr = config['network3cr']

            # construct graph for 3st network axial
            net_type3ax = config_net3ax['net_type']
            net_name3ax = config_net3ax['net_name']
            data_shape3ax = config_net3ax['data_shape']
            label_shape3ax = config_net3ax['label_shape']
            class_num3ax = config_net3ax['class_num']
            model_save_prefix3ax = config_net3ax['model_save_prefix'] + ".pt"

            full_data_shape3ax = [batch_size] + data_shape3ax
            # x3ax = tf.placeholder(tf.float32, shape = full_data_shape3ax)
            x3ax = torch.zeros(full_data_shape3ax, dtype=torch.float32, requires_grad=True)
            net_class3ax = NetFactory.create(net_type3ax)
            net3ax = net_class3ax(num_classes=class_num3ax, w_reg=None,
                                  b_reg=None, in_chns=full_data_shape3ax[1])
            # net3ax.set_params(config_net3ax)
            net3ax.load_state_dict(torch.load(model_save_prefix3ax, map_location=device)["model_state_dict"])
            predicty3ax = net3ax(x3ax)
            # proby3ax = tf.nn.softmax(predicty3ax)
            softmax3ax = torch.nn.Softmax(dim=-1)
            proby3ax = softmax3ax(predicty3ax)

            # construct graph for 3st network sagittal
            net_type3sg = config_net3sg['net_type']
            net_name3sg = config_net3sg['net_name']
            data_shape3sg = config_net3sg['data_shape']
            label_shape3sg = config_net3sg['label_shape']
            class_num3sg = config_net3sg['class_num']
            model_save_prefix3sg = config_net3sg['model_save_prefix'] + ".pt"

            # construct graph for 3st network
            full_data_shape3sg = [batch_size] + data_shape3sg
            # x3sg = tf.placeholder(tf.float32, shape = full_data_shape3sg)
            x3sg = torch.zeros(full_data_shape3sg, dtype=torch.float32, requires_grad=True)
            net_class3sg = NetFactory.create(net_type3sg)
            net3sg = net_class3sg(num_classes=class_num3sg, w_reg=None,
                                  b_reg=None, in_chns=full_data_shape3sg[1])
            # net3sg.set_params(config_net3sg)
            net3sg.load_state_dict(torch.load(model_save_prefix3sg, map_location=device)["model_state_dict"])
            predicty3sg = net3sg(x3sg)
            # proby3sg = tf.nn.softmax(predicty3sg)
            softmax3sg = torch.nn.Softmax(dim=-1)
            proby3sg = softmax3sg(predicty3sg)

            # construct graph for 3st network corogal
            net_type3cr = config_net3cr['net_type']
            net_name3cr = config_net3cr['net_name']
            data_shape3cr = config_net3cr['data_shape']
            label_shape3cr = config_net3cr['label_shape']
            class_num3cr = config_net3cr['class_num']
            model_save_prefix3cr = config_net3cr['model_save_prefix'] + ".pt"

            # construct graph for 3st network
            full_data_shape3cr = [batch_size] + data_shape3cr
            # x3cr = tf.placeholder(tf.float32, shape = full_data_shape3cr)
            x3cr = torch.zeros(full_data_shape3cr, dtype=torch.float32, requires_grad=True)
            net_class3cr = NetFactory.create(net_type3cr)
            net3cr = net_class3cr(num_classes=class_num3cr, w_reg=None,
                                  b_reg=None, in_chns=full_data_shape3cr[1])
            # net3cr.set_params(config_net3cr)
            net3cr.load_state_dict(torch.load(model_save_prefix3cr, map_location=device)["model_state_dict"])
            predicty3cr = net3cr(x3cr)
            softmax3cr = torch.nn.Softmax(dim=-1)
            # proby3cr = tf.nn.softmax(predicty3cr)
            proby3cr = softmax3cr(predicty3cr)

    # 3, create session and load trained models
    # all_vars = tf.global_variables()
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # if(config_net1):
    #     net1_vars = [x for x in all_vars if x.name[0:len(net_name1) + 1]==net_name1 + '/']
    #     saver1 = tf.train.Saver(net1_vars)
    #     saver1.restore(sess, config_net1['model_file'])
    # else:
    #     net1ax_vars = [x for x in all_vars if x.name[0:len(net_name1ax) + 1]==net_name1ax + '/']
    #     saver1ax = tf.train.Saver(net1ax_vars)
    #     saver1ax.restore(sess, config_net1ax['model_file'])
    #     net1sg_vars = [x for x in all_vars if x.name[0:len(net_name1sg) + 1]==net_name1sg + '/']
    #     saver1sg = tf.train.Saver(net1sg_vars)
    #     saver1sg.restore(sess, config_net1sg['model_file'])
    #     net1cr_vars = [x for x in all_vars if x.name[0:len(net_name1cr) + 1]==net_name1cr + '/']
    #     saver1cr = tf.train.Saver(net1cr_vars)
    #     saver1cr.restore(sess, config_net1cr['model_file'])
    #
    # if(config_test.get('whole_tumor_only', False) is False):
    #     if(config_net2):
    #         net2_vars = [x for x in all_vars if x.name[0:len(net_name2) + 1]==net_name2 + '/']
    #         saver2 = tf.train.Saver(net2_vars)
    #         saver2.restore(sess, config_net2['model_file'])
    #     else:
    #         net2ax_vars = [x for x in all_vars if x.name[0:len(net_name2ax)+1]==net_name2ax + '/']
    #         saver2ax = tf.train.Saver(net2ax_vars)
    #         saver2ax.restore(sess, config_net2ax['model_file'])
    #         net2sg_vars = [x for x in all_vars if x.name[0:len(net_name2sg)+1]==net_name2sg + '/']
    #         saver2sg = tf.train.Saver(net2sg_vars)
    #         saver2sg.restore(sess, config_net2sg['model_file'])
    #         net2cr_vars = [x for x in all_vars if x.name[0:len(net_name2cr)+1]==net_name2cr + '/']
    #         saver2cr = tf.train.Saver(net2cr_vars)
    #         saver2cr.restore(sess, config_net2cr['model_file'])
    #
    #     if(config_net3):
    #         net3_vars = [x for x in all_vars if x.name[0:len(net_name3) + 1]==net_name3 + '/']
    #         saver3 = tf.train.Saver(net3_vars)
    #         saver3.restore(sess, config_net3['model_file'])
    #     else:
    #         net3ax_vars = [x for x in all_vars if x.name[0:len(net_name3ax) + 1]==net_name3ax+ '/']
    #         saver3ax = tf.train.Saver(net3ax_vars)
    #         saver3ax.restore(sess, config_net3ax['model_file'])
    #         net3sg_vars = [x for x in all_vars if x.name[0:len(net_name3sg) + 1]==net_name3sg+ '/']
    #         saver3sg = tf.train.Saver(net3sg_vars)
    #         saver3sg.restore(sess, config_net3sg['model_file'])
    #         net3cr_vars = [x for x in all_vars if x.name[0:len(net_name3cr) + 1]==net_name3cr+ '/']
    #         saver3cr = tf.train.Saver(net3cr_vars)
    #         saver3cr.restore(sess, config_net3cr['model_file'])

    # 4, load test images
    import os
    print(os.curdir)
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    image_num = dataloader.get_total_image_number()

    # 5, start to test
    test_slice_direction = config_test.get('test_slice_direction', 'all')
    save_folder = config_data['save_folder']
    test_time = []
    struct = ndimage.generate_binary_structure(3, 2)
    margin = config_test.get('roi_patch_margin', 5)

    for i in range(image_num):

        # temp_imgs[155, h, w, 4]
        [temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
        t0 = time.time()
        # 5.1, test of 1st network
        data_shape = None
        if (config_net1):
            # data_shape [19, 180, 160, 4]
            data_shapes = [data_shape1[:-1], data_shape1[:-1], data_shape1[:-1]]
            data_shape = data_shape1
            # label shape [11, 180, 160, 1]
            label_shapes = [label_shape1[:-1], label_shape1[:-1], label_shape1[:-1]]
            nets = [net1, net1, net1]
            outputs = [proby1, proby1, proby1]
            inputs = [x1, x1, x1]
            class_num = class_num1
        else:
            data_shapes = [data_shape1ax[:-1], data_shape1sg[:-1], data_shape1cr[:-1]]
            data_shape = data_shape1ax[-1]
            label_shapes = [label_shape1ax[:-1], label_shape1sg[:-1], label_shape1cr[:-1]]
            nets = [net1ax, net1sg, net1cr]
            outputs = [proby1ax, proby1sg, proby1cr]
            inputs = [x1ax, x1sg, x1cr]
            class_num = class_num1ax

        # prob1 shape []
        prob1 = test_one_image_three_nets_adaptive_shape(temp_imgs, data_shapes, label_shapes, data_shape[-1],
                                                         class_num,
                                                         batch_size, nets, outputs, inputs, shape_mode=2)
        print("Prob1 shape is: " + str(prob1.shape))
        pred1 = np.asarray(np.argmax(prob1, axis=3), np.uint16)
        pred1 = pred1 * temp_weight

        wt_threshold = 2000
        if (config_test.get('whole_tumor_only', False) is True):
>>>>>>> 274ab01152d44901b354636771916dfe9b23eeca
            pred1_lc = ndimage.morphology.binary_closing(pred1, structure=struct)
            pred1_lc = get_largest_two_component(pred1_lc, False, wt_threshold)
            bbox1 = get_ND_bounding_box(pred1_lc, margin)
        sub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox1[0], bbox1[1]) for one_img in temp_imgs]
        sub_weight = crop_ND_volume_with_bounding_box(temp_weight, bbox1[0], bbox1[1])
    
        if (config_net2):
            data_shapes = [data_shape2[:-1], data_shape2[:-1], data_shape2[:-1]]
            label_shapes = [label_shape2[:-1], label_shape2[:-1], label_shape2[:-1]]
            nets = [net2, net2, net2]
            #inputs = [x2, x2, x2]
            class_num = class_num2
        else:
            data_shapes = [data_shape2ax[:-1], data_shape2sg[:-1], data_shape2cr[:-1]]
            label_shapes = [label_shape2ax[:-1], label_shape2sg[:-1], label_shape2cr[:-1]]
            nets = [net2ax, net2sg, net2cr]
            #inputs = [x2ax, x2sg, x2cr]
            class_num = class_num2ax
        for i in range(len(nets)):
            nets[i] = nets[i].to(device)
        prob2 = test_one_image_three_nets_adaptive_shape(sub_imgs, data_shapes, label_shapes, data_shape2ax[-1],
                                                         class_num, batch_size, nets, shape_mode=1)
        print("prob2 size: " + str(prob2.shape))
        
        # ================== CRF ==================================
        if WITH_CRF:
            d, h, w, c = prob2.shape
            prob2 = prob2.reshape(c, d, h, w)
            prob2 = np.expand_dims(prob2, axis=0)
            prob2 = crf_model(torch.tensor(prob2, dtype=torch.float32))
            print("prob2 size after crf: " + str(prob2.shape))

            prob2 = prob2.detach().numpy()[0].reshape(d, h, w, c)
        # ===========================================================
        pred2 = np.asarray(np.argmax(prob2, axis=3), np.uint16)
        pred2 = pred2 * sub_weight
        print("pred2 size" + str(pred2.shape))
        # ================================== End of 2nd network ==================================
    



        # ================================== test of 3rd network ==================================
        if (pred2.sum() == 0):
            [roid, roih, roiw] = sub_imgs[0].shape
            bbox2 = [[0, 0, 0], [roid - 1, roih - 1, roiw - 1]]
            subsub_imgs = sub_imgs
            subsub_weight = sub_weight
        else:
            pred2_lc = ndimage.morphology.binary_closing(pred2, structure=struct)
            pred2_lc = get_largest_two_component(pred2_lc)
            bbox2 = get_ND_bounding_box(pred2_lc, margin)
            subsub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox2[0], bbox2[1]) for one_img in sub_imgs]
            subsub_weight = crop_ND_volume_with_bounding_box(sub_weight, bbox2[0], bbox2[1])
    
        if (config_net3):
            data_shapes = [data_shape3[:-1], data_shape3[:-1], data_shape3[:-1]]
            label_shapes = [label_shape3[:-1], label_shape3[:-1], label_shape3[:-1]]
            nets = [net3, net3, net3]
            #inputs = [x3, x3, x3]
            class_num = class_num3
        else:
            data_shapes = [data_shape3ax[:-1], data_shape3sg[:-1], data_shape3cr[:-1]]
            label_shapes = [label_shape3ax[:-1], label_shape3sg[:-1], label_shape3cr[:-1]]
            nets = [net3ax, net3sg, net3cr]
            #inputs = [x3ax, x3sg, x3cr]
            class_num = class_num3ax
        for i in range(len(nets)):
            nets[i] = nets[i].to(device)
        prob3 = test_one_image_three_nets_adaptive_shape(subsub_imgs, data_shapes, label_shapes, data_shape3ax[-1],
                                                         class_num, batch_size, nets,
                                                         shape_mode=1)
        print("prob3 size: " + str(prob3.shape))
        
        # ================== CRF ==================================
        if WITH_CRF:
            d, h, w, c = prob3.shape
            prob3 = prob3.reshape(c, d, h, w)
            prob3 = np.expand_dims(prob3, axis=0)
            prob3 = crf_model(torch.tensor(prob3, dtype=torch.float32))
            print("prob3 size after crf: " + str(prob3.shape))
            prob3 = prob3.detach().numpy()[0].reshape(d, h, w, c)
        # ===========================================================
        pred3 = np.asarray(np.argmax(prob3, axis=3), np.uint16)
        pred3 = pred3 * subsub_weight
        print("pred3 size" + str(pred3.shape))
        # ================================== End of 3rd network ==================================



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


    # print(pred1.sum())
    test_time.append(time.time() - t0)
    final_label = np.zeros(temp_size, np.int16)
    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)
    final_labels.append((temp_name, final_label))
    print("final_label: " + str(final_label.shape))
    print("\n")

    save_array_as_nifty_volume(final_label, save_folder + "/{0:}.nii.gz".format(temp_name), img_names[0])
    