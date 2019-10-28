#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

from config import cfg


class VFELayer(object):
    def __init__(self,out_channels,name):
        super(VFELayer,self).__init__()
        self.units=int(out_channels/2)
        self.dense=tf.keras.layers.Dense(self.units,activation=tf.nn.relu,name='dense')
        self.batch_norm=tf.keras.layers.BatchNormalization(name="batch_norm")#fused=True)
    def __call__(self,inputs,mask,training):
        dense = self.dense(inputs)
        #[K,T,units]
        pointwise = self.batch_norm(dense)
        #[K,1,units]
        aggregated = tf.reduce_max(input_tensor=pointwise,axis=1,keepdims=True)
        #[K,T,units]
        repeated = tf.tile(aggregated,[1,cfg.VOXEL_POINT_COUNT,1])
        #[K,T,2*units]
        concatenated = tf.concat([pointwise,repeated],axis=2)

        mask = tf.tile(mask,[1,1,2*self.units])

        concatenated = tf.multiply(concatenated,tf.cast(mask,tf.float32))

        return concatenated

#voxel feature encoding
class VFELayer(object):
    def __init__(self,out_channels,name):
        super(VFELayer,self).__init__()
        self.units=int(out_channels/2)
        self.dense=tf.keras.layers.Dense(self.units,activation=tf.nn.relu,name='dense')
        self.batch_norm=tf.keras.layers.BatchNormalization(name="batch_norm")#fused=True)
    def __call__(self,inputs,mask,training):
        dense = self.dense(inputs)
        #[K,T,units]
        pointwise = self.batch_norm(dense)
        #[K,1,units]
        aggregated = tf.reduce_max(input_tensor=pointwise,axis=1,keepdims=True)
        #[K,T,units]
        repeated = tf.tile(aggregated,[1,cfg.VOXEL_POINT_COUNT,1])
        #[K,T,2*units]
        concatenated = tf.concat([pointwise,repeated],axis=2)

        mask = tf.tile(mask,[1,1,2*self.units])

        concatenated = tf.multiply(concatenated,tf.cast(mask,tf.float32))

        return concatenated

class FeatureNet(object):
    def __init__(self,input_feature,coordinate,training,batch_size,name=''):
        super(FeatureNet,self).__init__()
        self.training=training
        self.feature=input_feature
        #scalar
        self.batch_size=batch_size
        #self.feature

        #self.number  # To Do: import number

        # [ΣK, 4], each row stores (batch, d, h, w)
        self.coordinate = tf.convert_to_tensor(coordinate,name="coordinate")
        self.vfe1=VFELayer(32,'VFE-1')
        self.vfe2=VFELayer(128,'VFE-2')

        # boolean mask [K,T,2*units]
        # elimate the empty voxel
        mask = tf.not_equal(tf.reduce_max(input_tensor=self.feature,axis=2,keepdims=True),0)
        x = self.vfe1(self.feature,mask,self.training)
        x = self.vfe2(x,mask,self.training)
        # [K,128]
        voxelwise = tf.reduce_max(input_tensor=x,axis=1)
        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])


