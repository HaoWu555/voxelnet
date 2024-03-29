import sys
import os
import tensorflow as tf
import cv2
from numba import jit
from config import cfg
from utils import *
from model.group_pointcloud import FeatureNet
from model.rpn import MiddleAndRPN

class RPN3D(tf.keras.Model):
    def __init__(self,cls="Car",batch_size=1,learning_rate=0.001,max_gradient_norm=5.0,
                 alpha=1.5,beta=1,training=True):
        super(RPN3D,self).__init__()
        self.cls = cls
        self.batch_size = batch_size
        #self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        #self.global_step = 1
        self.epoch = tf.Variable(0, trainable=False)
        #self.epoch = 0
        #self.epoch_add_op = self.epoch.assign(self.epoch + 1)
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
        self.trainable_params=[]
        #self.delta_output = []
        #self.prob_output = []
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        #self.gradient_norm = []
        #self.tower_grads = []
        self.rpn = MiddleAndRPN()
        self.feature = FeatureNet()

        self.ckpt = tf.train.Checkpoint(step=self.global_step,optimizer=self.opt, model=self)


    @tf.function
    def train_step(self,voxel_feature,vox_number,voxel_coordinate,pos_equal_one,neg_equal_one,targets,pos_equal_one_for_reg,pos_equal_one_sum,neg_equal_one_sum,is_summary=False):
        # model running
        with tf.GradientTape() as tape:
            # feature is initial address for saving class FeatureNet
            self.feature(input_feature=voxel_feature,coordinate=voxel_coordinate,training=self.is_train,batch_size=self.batch_size)
            self.rpn(inputs=self.feature.outputs,pos_equal_one=pos_equal_one,neg_equal_one=neg_equal_one,targets=targets,pos_equal_one_for_reg=pos_equal_one_for_reg,pos_equal_one_sum=pos_equal_one_sum,neg_equal_one_sum=neg_equal_one_sum,alpha=self.alpha, beta=self.beta, training=self.is_train)

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

        ret=[]
        ret.extend((self.loss, self.reg_loss,self.cls_loss,self.cls_pos_loss,self.cls_neg_loss))

        return ret

    def save_model(self,save_model_path):
        #ckpt = tf.train.Checkpoint(step=self.global_step,optimizer=self.opt, net=self)
        manager = tf.train.CheckpointManager(self.ckpt,save_model_path,max_to_keep=3,checkpoint_name="RPN3D")
        save_path=manager.save()
        #print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("Saved checkpoint: {}".format(save_path))
        print("Saved step: {}".format(self.ckpt.step.numpy()))


    def restore_model(self,load_model_path):
        #ckpt = tf.train.Checkpoint(step=self.global_step,optimizer=self.opt, net=self)
        manager = tf.train.CheckpointManager(self.ckpt,load_model_path,max_to_keep=3)
        #print("Checkpoint: {}".format(manager.checkpoints))
        print("Last_Checkpoint: {}".format(manager.latest_checkpoint))
        status=self.ckpt.restore(manager.latest_checkpoint)
        #status.assert_consumed()
        #self.ckpt.restore(manager.checkpoints[0])
        print("Restore from global_step:{}".format(self.ckpt.step.numpy()))
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
            #print("Restored from {}".format(manager.checkpoints[0]))
        else:
            print("Initializing from scratch.")

        #print("{}:{}".format("loss",self.loss))
        #print("{}:{}".format("reg_loss",self.reg_loss))
        #print("{}:{}".format("cls_loss",self.cls_loss))
