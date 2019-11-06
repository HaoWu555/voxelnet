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


class RPN3D(tf.keras.Model):
    def __init__(self,cls="Car",batch_size=1,learning_rate=0.001,max_gradient_norm=5.0,
                 alpha=1.5,beta=1,training=True):
        super(RPN3D,self).__init__()
        self.cls = cls
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.alpha = alpha
        self.max_gradient_norm = max_gradient_norm
        self.beta = beta

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

        #self.delta_output = []
        #self.prob_output = []
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        #self.gradient_norm = []
        #self.tower_grads = []
        self.anchors = cal_anchors()
        self.rpn = MiddleAndRPN()
        self.feature = FeatureNet()

    #@tf.function
    def train_step(self,batch,is_summary=False):
            tag = batch[0]
            labels = batch[1]
            voxel_feature = batch[2]
            vox_number = tf.cast(batch[3],dtype=tf.int32) # original is int64
            voxel_coordinate =tf.cast(batch[4],dtype=tf.int32) # original is int64
            rgb = batch[5]
            raw_lidar = batch[6]
            # get from labels
            pos_equal_one, neg_equal_one, targets = cal_rpn_target(labels, [cfg.FEATURE_HEIGHT,cfg.FEATURE_WIDTH], self.anchors,cls=cfg.DETECT_OBJ,coordinate="lidar")

            pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
            pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
            neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
            info={"pos_equal_one":pos_equal_one,"neg_equal_one":neg_equal_one,"targets":tf.cast(targets,dtype=tf.float32),
                  "pos_equal_one_for_reg":pos_equal_one_for_reg,"pos_equal_one_sum":pos_equal_one_sum,
                  "neg_equal_one_sum":neg_equal_one_sum}

            print(tag)

            # model running
            with tf.GradientTape() as tape:
                # feature is initial address for saving class FeatureNet
                self.feature(*voxel_feature,*voxel_coordinate,training=self.is_train,batch_size=self.batch_size)
                self.rpn(inputs=self.feature.outputs,info=info,alpha=self.alpha, beta=self.beta, training=self.is_train)

            #update the parametr
            # list add A + B
            #self.params = rpn.variables+feature.variables
            self.loss = self.rpn.loss
            self.trainable_params = self.rpn.trainable_variables+self.feature.trainable_variables

            gradients = tape.gradient(self.loss,self.trainable_params)
            # clip the gradients
            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.opt.apply_gradients(zip(clipped_gradients,self.trainable_params))

            # output
            self.feature_output = self.feature.outputs
            self.delta_output=self.rpn.delta_output
            self.prob_output=self.rpn.prob_output

            # loss and grad
            self.reg_loss = self.rpn.reg_loss
            self.cls_loss = self.rpn.cls_loss
            self.cls_pos_loss = self.rpn.cls_pos_loss_rec
            self.cls_neg_loss = self.rpn.cls_neg_loss_rec

            self.rpn_output_shape= self.rpn.output_shapes

            if is_summary:
                tf.summary.scalar("loss",self.loss,step=self.opt.iterations)
                tf.summary.scalar("reg_loss",self.reg_loss,step=self.opt.iterations)
                tf.summary.scalar("cls_loss",self.cls_loss,step=self.opt.iterations)


            #print("{}:{}".format("loss",self.loss))
            #print("{}:{}".format("reg_loss",self.reg_loss))
            #print("{}:{}".format("cls_loss",self.cls_loss))




if __name__ == '__main__':
    train_summary_writer = tf.summary.create_file_writer('./summaries/train/')
    tf.random.set_seed(1)
    batch_size=2
    max_epoch = 2
    model=RPN3D(batch_size=batch_size, alpha=1.0,beta=10.0,training=False) # training only for bn and let it run in training mode but not inference mode
    batch_time = time.time()
    counter=0
    summary_interval = 1
    for epoch in range(max_epoch):
        for idx, batch in enumerate(iterate_data(train_dir,shuffle=True,aug=True,batch_size=batch_size,multi_gpu_sum=1,is_testset=False)):
            #if idx==10:
            #    break
            #else:
            print(idx)
            counter += 1
            start_time = time.time()

            if counter % summary_interval == 0:
                is_summary = True
            else:
                is_summary = False

            with train_summary_writer.as_default():
                model.train_step(batch,is_summary=is_summary)
            forward_time = time.time() - start_time
            batch_time = time.time() - batch_time

            print('train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f} '.format(counter,epoch, max_epoch, model.loss, model.reg_loss, model.cls_loss, model.cls_pos_loss, model.cls_neg_loss, forward_time, batch_time))

            with open('log/train.txt', 'a') as f:
                f.write( 'train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f} \n'.format(counter,epoch, max_epoch, model.loss, model.reg_loss, model.cls_loss, model.cls_pos_loss, model.cls_neg_loss, forward_time, batch_time))
                #batches = iterate_data(train_dir)
                #batch = next(batches)
        # save model
        #model.save('/save_model/',save_format='tf')
