# Braindon

# Project background

Project background was introduced in previous [README.md](background.md).


# Data

The data is available [here](https://utoronto-my.sharepoint.com/:u:/g/personal/yongzhao_wu_mail_utoronto_ca/ESqDxzWXkY5MoC3TDEliohQB3DCVO01rNCgNhstAb4lbRA?e=D0Zvw8). Make sure you've logged in to U of T account.

The data contains all BraTS 2017, with 10 samples from BraTS 2020 as test set.

**Note 1**: The name of folders in the data shows "BraTS 2018", that's because the official suggested us to use BraTS 2018 when we sending inquires regrading downloading BraTS 2017. They also confirmed that BraTS 2018 is identical with BraTS 2017.

[Image here]

**Note 2**: The data will be 24 GB after unzip.

# How to run the code?

## 1. Update config files to match your setup

If you are going to use the model to test/train with some examples, update ``data_root`` and ``data_names`` in config files in ``config17``. 

``data_root`` should be an absolute path to the data directory (so that there will be 3 folders under the path, HGG, LGG and 2020). 

``data_names`` should be an absolute path to the test data names. In our case, it should be ``test_names.txt``  or ``train_names.txt``, depends on your need, under folder ``config17``.  


## 2. Install required packages

All packages are in ``requirements.txt`` and available on ``pip``. 

## 3. Run the model

Training
-----

There are 3 tasks, and 3 views for each of them, so 9 models in total. 

For example, if you want to train model for *whole tumor* with *axial* view, then you can run the following line:

```
python train.py config17/train_wt_ax.txt
```

Testing
----- 


The following section will demonstrate how to run brain tumor segmentation with our model. After this section, segmentation result in ``.nii.gz`` format in ``result17``.

Testing Setup
-----
First, you have to **update** the model directory( ``model_file``) in ``config17/test_all.txt``. 

In our repo, we provided 18 models in two types:

 trained with **augmented** data and trained with **original** data.

 Models name that ends with ``_da`` are the models trained with **augmented** data, and others are trained with original data.

 Also, make sure that directory **result17** exists under project root folder.

Testing Model
------

 After all configs are done, the model can be test **with CRF** with following line:

 ```
 python test.py config17/test_all.txt crf
 ```

 If you **don't want CRF** in your test result, run the following line:

 ```
 python test.py config17/test_all.txt
 ```

Evaluate Model
-----

If you want to evaluate the accuracy of the model, make sure you have **completed the testing** step above.

The evaluation can be simply done with the following line:

```
 python evaluation.py config17/test_all.txt
```

The accuracy will be printed.