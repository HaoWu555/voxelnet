import os
import time
import sys
import tensorflow as tf
from itertools import count
import argparse
from config import cfg
from easydict import EasyDict
from utils.kitti_loader import iterate_data
from model.group_pointcloud import FeatureNet
import numpy as np
from model.rpn_1 import MiddleAndRPN
from utils.utils import *

#parser = argparse.ArgumentParser(description='training')
#parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=160,
#                    help='max epoch')
#parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
#                    help='set log tag')
#parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=2,
#                    help='set batch size')
#parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
#                    help='set learning rate')
#parser.add_argument('-al', '--alpha', type=float, nargs='?', default=1.0,
#                    help='set alpha in los function')
#parser.add_argument('-be', '--beta', type=float, nargs='?', default=10.0,
#                    help='set beta in los function')
#parser.add_argument('--output-path', type=str, nargs='?',
#                    default='./predictions', help='results output dir')
#parser.add_argument('-v', '--vis', type=bool, nargs='?', default=False,
#                    help='set the flag to True if dumping visualizations')
#args = parser.parse_args()

parser=EasyDict()
parser.i = 1
parser.tag = 'Test1'
parser.single_batch_size = 2
parser.lr =0.001
parser.al =1
parser.output_path = './prediction'
parser.v=False

dataset_dir = cfg.DATA_DIR
train_dir = os.path.join(cfg.DATA_DIR, 'training')
val_dir = os.path.join(cfg.DATA_DIR, 'validation')
log_dir = os.path.join('./log', parser.tag)
save_model_dir = os.path.join('./save_model', parser.tag)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)

batches=iterate_data(train_dir,batch_size=2)
#tag,labels,vox_feature,vox_number,vox_coordinate,rgb,raw_lidar
batch=next(batches)

anchors = cal_anchors()
pos_equal_one, neg_equal_one, targets = cal_rpn_target(batch[1], [cfg.FEATURE_HEIGHT,cfg.FEATURE_WIDTH], anchors)
pos_equal_one[..., [0]].shape
pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
info={"pos_equal_one":pos_equal_one,"neg_equal_one":neg_equal_one,"targets":tf.cast(targets,tf.float32),
      "pos_equal_one_for_reg":pos_equal_one_for_reg,"pos_equal_one_sum":pos_equal_one_sum,
      "neg_equal_one_sum":neg_equal_one_sum}

class RPN3D(object):
    def __init__(self,cls="Car",single_batch_size=1,learning_rate=0.001,max_gradient_norm=5.0,
                 alpha=1.5,beta=1,training=False,avail_gpus=['0']):
        self.cls = cls
        self.single_batch_size = single_batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.alpha = alpha
        self.beta = beta
        self.avail_gpus = avail_gpus

        boundaries = [80, 120]
        values = [ self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01 ]
        self.lr = tf.compat.v1.train.piecewise_constant(self.epoch, boundaries, values)

        #build graph
        self.is_train=training

        self.vox_feature = []
        self.vox_number = []
        self.vox_coordinate = []
        self.targets = []
        self.pos_equal_one = []
        self.pos_equal_one_sum = []
        self.pos_equal_one_for_reg = []
        self.neg_equal_one = []
        self.neg_equal_one_sum = []

        self.delta_output = []
        self.prob_output = []
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.gradient_norm = []
        self.tower_grads = []
        self.rpn = MiddleAndRPN()

    def __call__(self,a):
        print(a)

        for idx, dev in enumerate(self.avail_gpus):
            if idx==2:
                break
            else:
                batches = iterate_data(train_dir)
                batch = next(batches)
                tag = batch[0]
                labels = batch[1]
                voxel_feature = batch[2]
                vox_number = tf.cast(batch[3],dtype=tf.int32) # original is int64
                voxel_coordinate =tf.cast(batch[4],dtype=tf.int32) # original is int64
                rgb = batch[5]
                raw_lidar = batch[6]
                # get from labels
                pos_equal_one, neg_equal_one, targets = cal_rpn_target(labels, [cfg.FEATURE_HEIGHT,cfg.FEATURE_WIDTH], anchors)
                pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
                pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
                neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
                info={"pos_equal_one":pos_equal_one,"neg_equal_one":neg_equal_one,"targets":tf.cast(targets,dtype=tf.float32),
                      "pos_equal_one_for_reg":pos_equal_one_for_reg,"pos_equal_one_sum":pos_equal_one_sum,
                      "neg_equal_one_sum":neg_equal_one_sum}
                # model running
                with tf.GradientTape() as tape:
                    # feature is initial address for saving class FeatureNet
                    feature=FeatureNet(voxel_feature[0],voxel_coordinate[0],training=True,batch_size=1)
                    rpn=self.rpn(inputs=feature.outputs,info=info,alpha=self.alpha, beta=self.beta, training=self.is_train)
                feature_variables = feature.variables
                rpn_variables =rpn.variables

                # list add A + B
                print(len(rpn_variables+feature_variables))
                print(rpn.loss)
                gradients = tape.gradient(rpn.loss,feature_variables)
                #self.opt.apply_gradient()
                print(gradients)

                # input
                self.vox_feature.append(voxel_feature)
                self.vox_number.append(vox_number)
                self.vox_coordinate.append(feature.coordinate)
                self.targets.append(targets)
                self.pos_equal_one.append(pos_equal_one)
                self.pos_equal_one_sum.append(pos_equal_one_sum)
                self.pos_equal_one_for_reg.append(pos_equal_one_for_reg)
                self.neg_equal_one.append(neg_equal_one)
                self.neg_equal_one_sum.append(neg_equal_one_sum)

                # output
                feature_output = feature.outputs
                delta_output = rpn.delta_output
                prob_output = rpn.prob_output

                # loss and grad
                self.loss = rpn.loss
                self.reg_loss = rpn.reg_loss
                self.cls_loss = rpn.cls_loss
                self.cls_pos_loss = rpn.cls_pos_loss_rec
                self.cls_neg_loss = rpn.cls_neg_loss_rec


if __name__ == '__main__':
    model=RPN3D()
    model(1)
