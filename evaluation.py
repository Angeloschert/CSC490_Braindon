# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import os
import sys
sys.path.append('./')
import numpy as np
from util.data_process import load_3d_volume_as_array, binary_dice3d
from util.parse_config import parse_config

def get_ground_truth_names(g_folder, patient_names_file):
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    full_gt_names = []
    for patient_name in patient_names:
        patient_dir = os.path.join(g_folder, patient_name)
        img_names   = os.listdir(patient_dir)
        gt_name = None
        for img_name in img_names:
          
              if 'seg.' in img_name:
                  gt_name = img_name
                  break
        gt_name = os.path.join(patient_dir, gt_name)
        full_gt_names.append(gt_name)
    return full_gt_names

def get_segmentation_names(seg_folder, patient_names_file):
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    full_seg_names = []
    for patient_name in patient_names:
        seg_name = os.path.join(seg_folder, patient_name + '.nii.gz')
        full_seg_names.append(seg_name)
    return full_seg_names

def dice_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    dice_all_data = []
    for i in range(len(gt_names)):
        try:
            g_volume = load_3d_volume_as_array(gt_names[i])
            s_volume = load_3d_volume_as_array(seg_names[i])

            dice_one_volume = []
            if(type_idx ==0): # whole tumor
                temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
                dice_one_volume = [temp_dice]
            elif(type_idx == 1): # tumor core
                seg_=np.copy(s_volume)
                ground_=np.copy(g_volume)
                seg_[seg_==2]=0
                ground_[ground_==2]=0
                temp_dice = binary_dice3d(seg_ > 0, ground_ > 0)
                dice_one_volume = [temp_dice]
            else: #enhenced tumor
                temp_dice = binary_dice3d(s_volume ==4, g_volume ==4)
                dice_one_volume = [temp_dice]
                
            dice_all_data.append(dice_one_volume)
        except Exception as e:
            print(e)
    return dice_all_data
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python evaluation.py config17/test_all_class.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))

    config = parse_config(config_file)
    config_data = config.get('data')
    s_folder = config_data['save_folder']

    g_folder = config_data['data_root']
    patient_names_file = config_data["data_names"]

    print("="*15,"Evaluating","="*15)

    test_types = ['whole','core', 'enhenced']
    gt_names  = get_ground_truth_names(g_folder, patient_names_file)
    seg_names = get_segmentation_names(s_folder, patient_names_file)
    for type_idx in range(3):
        dice = dice_of_brats_data_set(gt_names, seg_names, type_idx)
        dice = np.asarray(dice)
        dice_mean = dice.mean(axis = 0)
        dice_std  = dice.std(axis  = 0)
        test_type = test_types[type_idx]
        np.savetxt(s_folder + '/dice_{0:}.txt'.format(test_type), dice)
        np.savetxt(s_folder + '/dice_{0:}_mean.txt'.format(test_type), dice_mean)
        np.savetxt(s_folder + '/dice_{0:}_std.txt'.format(test_type), dice_std)
        print('tissue type', test_type)
        print('dice mean  ', dice_mean)
        print('dice std   ', dice_std)
 
