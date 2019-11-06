import os
import time
import sys
import glob
import tensorflow as tf
from itertools import count
import argparse
from config import cfg
from easydict import EasyDict
from utils.kitti_loader import iterate_data,sample_test_data
import numpy as np
from utils.utils import *
from model.model import RPN3D
#from train_hook import check_if_should_pause

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=160,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='Test1',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=2,
                    help='set batch size')
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                    help='set learning rate')
parser.add_argument('-al', '--alpha', type=float, nargs='?', default=1.0,
                    help='set alpha in los function')
parser.add_argument('-be', '--beta', type=float, nargs='?', default=10.0,
                    help='set beta in los function')
parser.add_argument('--output-path', type=str, nargs='?',
                    default='./predictions', help='results output dir')
parser.add_argument('-v', '--vis', type=bool, nargs='?', default=False,
                    help='set the flag to True if dumping visualizations')
args = parser.parse_args()
print(args)

#parser=EasyDict()
#parser.i = 1
#parser.tag = 'Test1'
#parser.single_batch_size = 2
#parser.lr =0.001
#parser.al =1
#parser.output_path = './prediction'
#parser.v=False

dataset_dir = cfg.DATA_DIR
train_dir = os.path.join(cfg.DATA_DIR, 'training')
val_dir = os.path.join(cfg.DATA_DIR, 'validation')
log_dir = os.path.join('./log', args.tag)
save_model_dir = os.path.join('./save_model', args.tag)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)


if __name__ == '__main__':
    train_summary_writer = tf.summary.create_file_writer('./summaries/train/')
    tf.random.set_seed(1)

    batch_size=args.single_batch_size
    max_epoch = 1

    # training only for bn and let it run in training mode but not inference mode
    model=RPN3D(cls=cfg.DETECT_OBJ,batch_size=batch_size, alpha=1.0,beta=10.0,training=False)
    # save
    #model.save_weights(save_model_dir+'save_model_1')
    batch_time = time.time()
    counter=0
    summary_interval = 1
    anchors = cal_anchors()
    for epoch in range(max_epoch):
        for idx, batch in enumerate(iterate_data(train_dir,shuffle=True,aug=True,batch_size=batch_size,multi_gpu_sum=1,is_testset=False)):
            if idx==3:
                break

            print(idx)
            # get the data
            tag = batch[0]
            print(tag)
            labels = batch[1]
            voxel_feature = batch[2]
            vox_number = batch[3] # original is int64
            voxel_coordinate =batch[4] # original is int64
            rgb = batch[5]
            raw_lidar = batch[6]

            # get from labels
            pos_equal_one, neg_equal_one, targets = cal_rpn_target(labels, [cfg.FEATURE_HEIGHT,cfg.FEATURE_WIDTH], anchors,cls=cfg.DETECT_OBJ,coordinate="lidar")
            pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
            pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
            neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

            counter += 1
            start_time = time.time()

            if counter % summary_interval == 0:
                is_summary = True
            else:
                is_summary = False

            with train_summary_writer.as_default():
                model.train_step(voxel_feature=voxel_feature,vox_number=vox_number,voxel_coordinate=voxel_coordinate,pos_equal_one=pos_equal_one,neg_equal_one=neg_equal_one,targets=targets,pos_equal_one_for_reg=pos_equal_one_for_reg,pos_equal_one_sum=pos_equal_one_sum,neg_equal_one_sum=neg_equal_one_sum,is_summary=is_summary)
            forward_time = time.time() - start_time
            batch_time = time.time() - batch_time

            print('train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f} '.format(counter,epoch, max_epoch, model.loss, model.reg_loss, model.cls_loss, model.cls_pos_loss, model.cls_neg_loss, forward_time, batch_time))

            with open('log/train.txt', 'a') as f:
                f.write( 'train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f} \n'.format(counter,epoch, max_epoch, model.loss, model.reg_loss, model.cls_loss, model.cls_pos_loss, model.cls_neg_loss, forward_time, batch_time))

            #model.save_weights(save_model_dir)
                #batches = iterate_data(train_dir)
                #batch = next(batches)
        # save model
        #model.save('/save_model/',save_format='tf')
