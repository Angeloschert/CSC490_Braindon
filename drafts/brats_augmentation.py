# -*- coding: utf-8 -*-
"""brats_augmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TkReeDGf02yvLTnB2NOkAHfnzuiEgdxi
"""

#!pip install torch
#!pip install torchio
#!pip install SimpleITK

import torchio as tio
from os.path import join
import os
import SimpleITK as sitk
import numpy as np
import shutil
import random

def data_augment(data_path, num_transform_choice=2):
    sample_names = os.listdir(data_path)
    transforms = [
        # spatial
        tio.RandomAnisotropy(),
        tio.RandomAffine(),
        tio.RandomElasticDeformation(num_control_points=10),
        tio.RandomFlip(axes=['inferior-superior'], flip_probability=1),
    ]

    for image_folder in sample_names:
        print('Processing .... ', image_folder)
        file_names = os.listdir(join(data_path, image_folder))
        names = {}
        for i in file_names:
            if i.startswith('ROI'):
                names['ROI'] = join(data_path, image_folder, i)
            else:
                split = i.split('_')
                split2 = split[-1].split('.')
                names[split2[0]] = join(data_path, image_folder, i)
        subject_a = tio.Subject(
            t1=tio.ScalarImage(names['t1']),
            t1ce=tio.ScalarImage(names['t1ce']),
            t2=tio.ScalarImage(names['t2']),
            flair=tio.ScalarImage(names['flair']),
            ROI=tio.ScalarImage(names['ROI']),
            label=tio.LabelMap(names['seg'])
        )
        # transform = tio.RandomElasticDeformation(num_control_points=10)
        transform = tio.Compose(random.sample(transforms, num_transform_choice))
        new_subject = transform(subject_a)
        new_folder = 'augmentation_' + image_folder
        if not os.path.isdir(join(data_path, new_folder)):
            os.mkdir(join(data_path, new_folder))
        else:
            shutil.rmtree(join(data_path, new_folder))
            os.mkdir(join(data_path, new_folder))
            
        for i in new_subject:
            sitk.WriteImage(new_subject.get(i).as_sitk(), join(data_path, new_folder, 'sample_' + names[i if i != 'label' else 'seg'].split('/')[-1]))

if __name__ == "__main__":
    # change based on server settings
    from google.colab import drive
    drive.mount('/content/drive')

    LGG = 'drive/MyDrive/BRATS2017/Brats17TrainingData/LGG/'
    data_augment(LGG, 2)