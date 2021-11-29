# Braindon

# Project background

Project background was introduced in previous [README.md](background.md).

Further more, our project were based on this [this repo](https://github.com/taigw/brats17/), which won the second place in BraTS 2017. We follow the original model for the config setup and extraction, and we did minor modifications in ``util/``, and major/completely rework on other files.

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

There are two configs for testing in the ``config17``, which are ``config17/test_all.txt`` and ``config17/test_all_da.txt``, which are for original data and augmented data, respectively. 


 Also, make sure that directory **result17** exists under project root folder.

Testing Model
------

As mentioned above, there are two configs, the following examples will be using original data. Change ``test_all.txt`` to ``test_all_da.txt`` if you want to test with model trained with augmented data. 

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


# 4. Result

The performance for the model in testing set are the following:

<table>
   <tr>
      <td></td>
      <td>Original</td>
      <td></td>
      <td></td>
      <td>Augmented</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td>ET</td>
      <td>WT</td>
      <td>TC</td>
      <td>ET</td>
      <td>WT</td>
      <td>TC</td>
   </tr>
   <tr>
      <td>Mean</td>
      <td>68.15</td>
      <td>90.11</td>
      <td>81.49</td>
      <td>76.38</td>
      <td>88.93</td>
      <td>81.68</td>
   </tr>
   <tr>
      <td>STD</td>
      <td>30.77</td>
      <td>6.28</td>
      <td>16.64</td>
      <td>25.93</td>
      <td>8.88</td>
      <td>13.24</td>
   </tr>
   <tr>
      <td></td>
   </tr>
</table>