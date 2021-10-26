from __future__ import absolute_import, print_function
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_chns, out_chns, kernels=[[3, 3, 1], [3, 3, 1]], strides=[[1,1,1], [1,1,1]], dilation_rate=[[1,1,1], [1,1,1]], activation=None, w_init=None, w_reg=None, res=True):
        super().__init__()
        self.in_chns = in_chns
        self.out_chns = out_chns
        self.kernels = kernels
        self.strides = strides
        self.dilation_rate = dilation_rate
        if not activation:
            self.activation = nn.PReLU()
        else:
            self.activation = activation
        self.w_init = w_init
        self.w_reg = w_reg
        self.res = res
        
    
    def forward(self, x):
        output = x
        #print(len(self.kernels))
        in_chns = self.in_chns
        for i in range(len(self.kernels)):
            kernel, stride, dilation = self.kernels[i], self.strides[i], self.dilation_rate[i]
            #print(self.in_chns, self.out_chns, kernel, stride, dilation)
            #print(output.shape)
            self.conv3d = nn.Conv3d(in_chns, self.out_chns, kernel_size=kernel, padding='same', dilation=dilation)
            self.batchnorm = nn.BatchNorm3d(in_chns)
            in_chns = self.out_chns
            output = self.batchnorm(output)
            output = self.activation(output)
            output = self.conv3d(output)
        if self.res:
            if self.in_chns != self.out_chns:
                self.projector = nn.Conv3d(self.in_chns, self.out_chns, kernel_size=1, stride=1, padding='same')
                x = self.projector(x)
            output += x
        #print("Finish block")
        return output
               
class Conv2dBlock(nn.Module):
    def __init__(self,in_chns, out_chns, kernels, padding=0, strides=[1, 1, 1], activation=nn.PReLU(), w_init=None, w_reg=None, b_init=None, b_reg=None, with_bn=True, deconv=False):
        super().__init__()
        self.in_chns = in_chns
        self.out_chns = out_chns
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.w_init = w_init
        self.w_reg = w_reg
        self.b_init = b_init
        self.b_reg = b_reg
        if not deconv:
            self.conv_block = nn.Conv3d(in_chns, out_chns, kernel_size=kernels, padding=padding, stride=strides, bias=True)
        else:
            self.conv_block = nn.ConvTranspose3d(in_chns, out_chns, kernel_size=kernels, padding=padding, stride=strides, bias=True)
            
        if with_bn:
            self.bn = nn.BatchNorm3d(self.out_chns)
        else:
            self.bn = nn.Identity()
    
    def forward(self, x):
        output = self.conv_block(x)
        output = self.bn(output)
        output = self.activation(output)
        return output
    
class SliceLayer(nn.Module):
    
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin
        
    def forward(self, x):
        # TODO: Fix this
        return x[:, :, :, :, self.margin:-(self.margin)]
    
class MSNet(nn.Module):

    def __init__(
        self,
        in_chns,
        num_classes,
        w_init=None,
        w_reg=None,
        b_init=None,
        b_reg=None,
        activation=nn.PReLU(),
    ):

        super().__init__()
        self.num_classes = num_classes
        self.w_init, self.w_reg, self.b_init, self.b_reg = w_init, w_reg, b_init, b_reg
        self.activation = activation
        self.base_chns = [32, 32, 32, 32]
        self.is_WTNet = True

        def init_weights(m):
            if isinstance(m, ResBlock) or isinstance(m, Conv2dBlock):
                try:
                  torch.nn.init.xavier_normal_(m.w_init)
                except:
                  print("No w_init")
                try:
                  torch.nn.init.xavier_normal_(m.b_init)
                except:
                  print("No b_init")

        # First Block

        self.block1 = nn.Sequential(
            ResBlock(
                in_chns,
                self.base_chns[0],
                activation=activation,
                w_init=w_init, w_reg=w_reg
            ),
            ResBlock(
                self.base_chns[0],
                self.base_chns[0],
                activation=activation,
                w_init=w_init, w_reg=w_reg
            )
        )
        # torch.nn.init.xavier_normal_(self.block1[1].weight)
        # torch.nn.init.xavier_normal_(self.block1[2].weight)
        self.block1.apply(init_weights)

        self.fuse1 = Conv2dBlock(
            self.base_chns[0],
            self.base_chns[0],
            kernels=[1, 1, 3],
            padding='valid',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.fuse1.apply(init_weights)

        self.downsample1 = Conv2dBlock(
            self.base_chns[0],
            self.base_chns[0],
            kernels=[3, 3, 1],
            strides=[2, 2, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.downsample1.apply(init_weights)

        self.feature_expand1 = Conv2dBlock(
            self.base_chns[0],
            self.base_chns[1],
            kernels=[1, 1, 1],
            strides=[1, 1, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.feature_expand1.apply(init_weights)

        # Second Block

        self.block2 = nn.Sequential(
            ResBlock(
                self.base_chns[1],
                self.base_chns[1],
                activation=activation,
                w_init=w_init, w_reg=w_reg
            ),
            ResBlock(
                self.base_chns[1],
                self.base_chns[1],
                activation=activation,
                w_init=w_init, w_reg=w_reg
            )
        )
        self.block2.apply(init_weights)

        self.fuse2 = Conv2dBlock(
            self.base_chns[1],
            self.base_chns[1],
            kernels=[3, 3, 1],
            padding='valid',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.fuse2.apply(init_weights)

        self.downsample2 = Conv2dBlock(
            self.base_chns[1],
            self.base_chns[1],
            kernels=[3, 3, 1],
            strides=[2, 2, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.downsample2.apply(init_weights)

        self.feature_expand2 = Conv2dBlock(
            self.base_chns[1],
            self.base_chns[2],
            kernels=[1, 1, 1],
            strides=[1, 1, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.feature_expand2.apply(init_weights)

        self.pred_1E = nn.Conv3d(
            self.base_chns[1], 
            self.num_classes,               
            kernel_size=[3, 3, 1], 
            padding='same'
        )

        self.pred_1WT = Conv2dBlock(
            self.base_chns[1],
            self.num_classes,
            kernels=[3, 3, 1],
            strides=[2, 2, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
            deconv=True,
        )
        self.pred_1WT.apply(init_weights)

        # Third Block

        self.block3 = nn.Sequential(
            ResBlock(
                self.base_chns[2],
                self.base_chns[2],
                dilation_rate=[[1, 1, 1], [1, 1, 1]],
                activation=activation,
                w_init=w_init,
                w_reg=w_reg,
            ), 
            ResBlock(
                self.base_chns[2],
                self.base_chns[2],
                strides=[[2, 2, 1], [2, 2, 1]],
                activation=activation,
                w_init=w_init,
                w_reg=w_reg,
            ), 
            ResBlock(
                self.base_chns[2],
                self.base_chns[2],
                dilation_rate=[[3, 3, 1], [3, 3, 1]],
                activation=activation,
                w_init=w_init,
                w_reg=w_reg,
            )
        )
        self.block3.apply(init_weights)

        self.fuse3 = Conv2dBlock(
            self.base_chns[2],
            self.base_chns[2],
            kernels=[1, 1, 3],
            padding='valid',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.fuse3.apply(init_weights)

        self.feature_expand3 = Conv2dBlock(
            self.base_chns[2],
            self.base_chns[3],
            kernels=[1, 1, 1],
            strides=[1, 1, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.feature_expand3.apply(init_weights)

        self.pred_21 = Conv2dBlock(
            self.base_chns[2],
            self.num_classes * 2,
            kernels=[3, 3, 1],
            strides=[2, 2, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
            deconv=True,
        )
        self.pred_21.apply(init_weights)

        self.pred_22 = Conv2dBlock(
            self.num_classes * 2,
            self.num_classes * 2,
            kernels=[3, 3, 1],
            strides=[2, 2, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
            deconv=True,
        )
        self.pred_22.apply(init_weights)

        # Fourth Block

        self.block4 = nn.Sequential(
            ResBlock(
                self.base_chns[3],
                self.base_chns[3],
                dilation_rate=[[3, 3, 1], [3, 3, 1]],
                activation=activation,
                w_init=w_init,
                w_reg=w_reg,
            ), 
            ResBlock(
                self.base_chns[3],
                self.base_chns[3],
                dilation_rate=[[2, 2, 1], [2, 2, 1]],
                activation=activation,
                w_init=w_init,
                w_reg=w_reg,
            ), 
            ResBlock(
                self.base_chns[3],
                self.base_chns[3],
                dilation_rate=[[1, 1, 1], [1, 1, 1]],
                activation=activation,
                w_init=w_init,
                w_reg=w_reg,
            )
        )
        self.block4.apply(init_weights)

        self.fuse4 = Conv2dBlock(
            self.base_chns[3],
            self.base_chns[3],
            kernels=[1, 1, 3],
            padding='valid',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
        )
        self.fuse4.apply(init_weights)

        self.pred_31 = Conv2dBlock(
            self.base_chns[3],
            self.num_classes * 4,
            kernels=[3, 3, 1],
            strides=[2, 2, 1],
            # padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
            deconv=True,
        )
        self.pred_31.apply(init_weights)

        self.pred_32 = Conv2dBlock(
            self.num_classes * 4,
            self.num_classes * 4,
            kernels=[3, 3, 1],
            strides=[2, 2, 1],
           #  padding='same',
            activation=self.activation,
            w_init=self.w_init,
            w_reg=self.w_reg,
            b_init=self.b_init,
            b_reg=self.b_reg,
            deconv=True,
        )
        self.pred_32.apply(init_weights)

        # TODO: Change this MAYBE

        self.final_pred = nn.Conv3d(14, self.num_classes, kernel_size=[3, 3, 1], padding='same')
        self.centra_slice1 = SliceLayer(margin=2)
        self.centra_slice2 = SliceLayer(margin=1)


    def forward(self, x):
        f1 = x
        # print("f1", f1.shape)
        f1 = self.block1(f1)
        # print("f1", f1.shape)
        f1 = self.fuse1(f1)
        # print("f1", f1.shape)
        if self.is_WTNet:
            f1 = self.downsample1(f1)
        if self.base_chns[0] != self.base_chns[1]:
            f1 = self.feature_expand1(f1)
        # print("f1", f1.shape)

        f1 = self.block2(f1)
        # print("f1", f1.shape)
        f1 = self.fuse2(f1)
        # print("f1", f1.shape)
        f2 = self.downsample2(f1)
        if self.base_chns[1] != self.base_chns[2]:
            f2 = self.feature_expand1(f2)
        f2 = self.block3(f2)
        f2 = self.fuse3(f2)
        # print("f2", f2.shape)
        
        f3 = f2
        if self.base_chns[2] != self.base_chns[3]:
            f3 = self.feature_expand1(f3)
        f3 = self.block4(f3)
        f3 = self.fuse3(f3)
        # print("f3", f3.shape)
        # Prediction

        p1 = self.centra_slice1(f1)
        # print(f1.shape, p1.shape)
        # print(p1.shape)
        if self.is_WTNet:
            p1 = self.pred_1WT(p1)
        else:
            p1 = self.pred_1E(p1)

        p2 = self.centra_slice2(f2)
        # print(f2.shape, p2.shape)
        p2 = self.pred_21(p2)
        if self.is_WTNet:
            p2 = self.pred_22(p2)

        p3 = self.pred_31(f3)
        if self.is_WTNet:
            p3 = self.pred_32(p3)

        # print(p1.shape, p2.shape, p3.shape)
        combine = torch.cat([p1, p2, p3], 1)
        # print(combine.shape)
        return self.final_pred(combine)