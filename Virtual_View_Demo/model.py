from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as layers
from util import *
arg_scope = tf.contrib.framework.arg_scope
monodepth_parameters = namedtuple('parameters',
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'synthesis_height, '
                        'synthesis_width, '
                        'demo_tum')

class MonodepthModel(object):
    def __init__(self,params,mode,reference=None,left=None,right=None,param_path=None,reuse_variables=None,model_index=0,now_scale=None):
        self.params = params
        self.mode = mode
        if mode == 'demo':
            self.is_training = False
        else:
            print('mode is undecleared')
            return

        ## cam_param
        self.param_path = param_path

        ## image
        if mode == 'demo':
            self.right = right
            self.left = left
            self.demo_tum = params.demo_tum
            self.now_scale = tf.reshape(now_scale,[1])

            self.left_intrinsic,self.left_pose, self.reference_intrinsic, self.reference_pose,self.right_intrinsic,self.right_pose = self.load_kocca_param_demo(self.param_path)
            self.reconstructed_image = []
        else:
            if reference==None:
                print('reference image is undecleared')
                return
            self.left = reference
            self.right = reference
            self.reference = reference

        self.model_collection = ['model_'+str(model_index)]

        self.reuse_variables = reuse_variables
        self.MAX_SCALE = 100
        self.MIN_DISP = 0.1
        self.epoch=False

        self.build_outputs()

    ################################################################################################
    ## param function
    def load_kocca_param_demo(self,path):
        l_k, l_quar, l_c, r_k, r_quar, r_c = tf.py_func(load_kocca_param_demo, [path],
                                                        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                                         tf.float32])
        num, side = path.get_shape().as_list()

        l_k = tf.reshape(l_k, [num, 3, 3])
        l_quar = tf.reshape(l_quar, [num, 1, 4])
        l_c = tf.reshape(l_c, [num, 3, 1])

        r_k = tf.reshape(r_k, [num, 3, 3])
        r_quar = tf.reshape(r_quar, [num, 1, 4])
        r_c = tf.reshape(r_c, [num, 3, 1])

        m_c = l_c + (r_c - l_c) * (tf.reshape(self.now_scale,[1]) - 1)
        m_quar = quar_interpolation(l_quar, r_quar, (tf.reshape(self.now_scale,[1]) - 1))
        m_k = l_k + (r_k - l_k) * (tf.reshape(self.now_scale,[1]) - 1)

        l_rot = quar_to_rot(l_quar)
        r_rot = quar_to_rot(r_quar)
        m_rot = quar_to_rot(m_quar)

        l_t = cal_t(l_rot, l_c)
        m_t = cal_t(m_rot, m_c)
        r_t = cal_t(r_rot, r_c)

        l_r = tf.concat([l_rot, l_t], 2)
        m_r = tf.concat([m_rot, m_t], 2)
        r_r = tf.concat([r_rot, r_t], 2)

        return l_k, l_r, m_k, m_r, r_k, r_r


    ####################################################################################
    # up resize function
    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])


    def get_disp_mono(self, x,scope,reuse):
        disp = self.conv(x, 1, 3, 1,scope,reuse,activation_fn= tf.nn.sigmoid) ## 0~0.3 값을 가짐 , shape 은 input=output, but channel = 2
        return disp

    def disp_to_depth(self, disp):
        min_disp = 1 / self.MAX_SCALE
        max_disp = 1 / self.MIN_DISP
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    ###############################################################################################
    # layer convenience function
    def conv(self, x, num_out_layers, kernel_size, stride,scope,reuse,activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]]) ## shape = (2, height,width,3)로 들어옴, height와 width에 양쪽으로 p만큼 padding을 하는 line
        re = slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=scope,reuse=reuse)
        return re

    def conv2d(self, x, num_out_layers, kernel_size, stride,scope,  reuse,activation_fn=None):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p],
                         [0, 0]])  ## shape = (2, height,width,3)로 들어옴, height와 width에 양쪽으로 p만큼 padding을 하는 line
        re = slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),scope=scope,reuse=reuse)

        return re

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]]) ## 여기서도 마찬가지로 height,width에만 padding이 들어가게끔 함
        return slim.max_pool2d(p_x, kernel_size)

    def resblock_basic(self,x,num_layer,num_blocks,scope,reuse,down_sample = True,is_training=False):
        for i in range(num_blocks):
            identity = x
            if down_sample == True:
                conv1 = self.conv2d(x,num_layer,3,2,scope+'/conv_{}_1'.format(i),reuse)
                identity = self.conv2d(identity,num_layer,1,2,scope+'/identity_{}'.format(i),reuse)
                identity = self.batch_norm(identity,scope+'/identity_batch_{}'.format(i),reuse,is_training)
                down_sample = False
            else:
                conv1 = self.conv2d(x,num_layer,3,1,scope+'/conv_{}_1'.format(i),reuse)
            bn1 = self.batch_norm(conv1,scope+'/batch_{}_1'.format(i),reuse,is_training)
            relu1 = tf.nn.relu(bn1)

            conv2 = self.conv2d(relu1,num_layer,3,1,scope+'/conv_{}_2'.format(i),reuse)
            bn2 = self.batch_norm(conv2,scope+'/batch_{}_2'.format(i),reuse,is_training)

            x = tf.nn.relu(bn2+identity)
        return x

    def batch_norm(self,x,scope,reuse,is_training=False):
        return tfc.layers.batch_norm(x,decay=0.1,epsilon=1e-05,scale=True,scope=scope,reuse=reuse,is_training=is_training)

    def upconv_mono2(self, x, num_out_layers, kernel_size, scale,scope,reuse):
        conv = self.conv(x, num_out_layers, kernel_size, 1,scope,reuse)
        upsample = self.upsample_nn(conv, scale)
        return upsample

    #########################################################################################
    # backbone
    def res_encoder(self,color,reuse):
        block = self.resblock_basic
        conv = self.conv2d
        with tf.variable_scope('encoder',reuse=reuse):
            conv1 = conv(color, 64, 7, 2,'conv1',reuse)
            relu = tf.nn.relu(conv1)
            pool1 = self.maxpool(relu, 3)
            res_block1 = block(pool1, 64, 2,'res_block1', reuse,down_sample=False, is_training=self.is_training)
            res_block2 = block(res_block1, 128, 2,'res_block2',reuse, down_sample=True, is_training=self.is_training)
            res_block3 = block(res_block2, 256, 2, 'res_block3',reuse,down_sample=True, is_training=self.is_training)
            res_block4 = block(res_block3, 512, 2,'res_block4',reuse, down_sample=True, is_training=self.is_training)

        return [relu,res_block1,res_block2,res_block3,res_block4]


    def depth_decoder(self,feature,reuse):
        skip1 = feature[0]
        skip2 = feature[1]
        skip3 = feature[2]
        skip4 = feature[3]
        skip5 = feature[4]

        upconv = self.upconv_mono2
        conv_elu = self.conv

        with tf.variable_scope('depth_decoder',reuse=reuse):
            upconv5 = upconv(skip5, 256, 3, 2,'upconv5',reuse)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv_elu(concat5, 256, 3, 1,'iconv5',reuse)

            upconv4 = upconv(iconv5, 128, 3, 2,'upconv4',reuse)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv_elu(concat4, 128, 3, 1,'iconv4',reuse)
            disp4 = self.get_disp_mono(iconv4,'disp4',reuse)

            upconv3 = upconv(iconv4, 64, 3, 2,'upconv3',reuse)
            concat3 = tf.concat([upconv3, skip2], 3)
            iconv3 = conv_elu(concat3, 64, 3, 1,'iconv3',reuse)
            disp3 = self.get_disp_mono(iconv3,'disp3',reuse)

            upconv2 = upconv(iconv3, 32, 3, 2,'upconv2',reuse)
            concat2 = tf.concat([upconv2, skip1], 3)
            iconv2 = conv_elu(concat2, 32, 3, 1,'iconv2',reuse)
            disp2 = self.get_disp_mono(iconv2,'disp2',reuse)

            upconv1 = upconv(iconv2, 16, 3, 2,'upconv1',reuse)
            iconv1 = conv_elu(upconv1, 16, 3, 1,'iconv1',reuse)
            disp1 = self.get_disp_mono(iconv1,'disp1',reuse)

            disp4, depth4 = self.disp_to_depth(self.upsample_nn(disp4, 8))
            disp3, depth3 = self.disp_to_depth(self.upsample_nn(disp3, 4))
            disp2, depth2 = self.disp_to_depth(self.upsample_nn(disp2, 2))
            disp1, depth1 = self.disp_to_depth(disp1)

        return [disp1,disp2,disp3,disp4],[depth1,depth2,depth3,depth4]

    def build_outputs(self):
        with tf.variable_scope('out_put'):
            ## feature
            if self.mode == 'demo':
                self.left_feature = self.res_encoder(
                    (tf.image.resize_bilinear(self.left, [self.params.height, self.params.width]) - 0.45) / 0.225,
                    self.reuse_variables)
                self.right_feature = self.res_encoder(
                    (tf.image.resize_bilinear(self.right, [self.params.height, self.params.width]) - 0.45) / 0.225,
                    True)
            else:
                self.reference_feature = self.res_encoder(self.reference, self.reuse_variables)

            ## decoder
            if self.mode == 'demo':
                self.disp_left_est, self.depth_left_est = self.depth_decoder(self.left_feature, self.reuse_variables)
                self.disp_right_est, self.depth_right_est = self.depth_decoder(self.right_feature, True)
                self.synthesis()
                return
            else:
                self.disp_reference_est, self.depth_reference_est = self.depth_decoder(self.reference_feature,
                                                                                       self.reuse_variables)
                return


    def synthesis(self):

        over_coords_l, warped_l = projective_depth_forward_warp(tf.image.resize_bilinear(self.depth_left_est[0],
                                                                                      [self.params.synthesis_height,
                                                                                       self.params.synthesis_width]),
                                                             self.left_pose, self.left_intrinsic,
                                                             self.reference_pose, self.reference_intrinsic)
        over_coords_r, warped_r = projective_depth_forward_warp(tf.image.resize_bilinear(self.depth_right_est[0],
                                                                                      [self.params.synthesis_height,
                                                                                       self.params.synthesis_width]),
                                                             self.right_pose, self.right_intrinsic,
                                                             self.reference_pose, self.reference_intrinsic)

        Im_l, coord_l = projective_inverse_warp(tf.image.resize_bilinear(self.left, [self.params.synthesis_height,
                                                                                          self.params.synthesis_width]),
                                                     warped_l, self.reference_pose, self.reference_intrinsic,
                                                     self.left_pose, self.left_intrinsic)

        Im_r, coord_r = projective_inverse_warp(tf.image.resize_bilinear(self.right, [self.params.synthesis_height,
                                                                                           self.params.synthesis_width]),
                                                     warped_r, self.reference_pose, self.reference_intrinsic,
                                                     self.right_pose, self.right_intrinsic)


        depth_mask1, depth_mask2, common_mask = depth_synthesis(warped_l,
                                                                warped_r, tf.logical_or(over_coords_l,coord_l),
                                                                tf.logical_or(over_coords_r,coord_r),
                                                                self.MAX_SCALE)

        recon, _ = synthesis_reconstruction(depth_mask1, depth_mask2, common_mask, Im_l, Im_r,
                                                      self.reference_pose, self.left_pose, self.right_pose,True)

        self.reconstructed_image.append(recon)

