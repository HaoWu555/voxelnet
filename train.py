import os
import time
import sys
import tensorflow as tf
from itertools import count

from config import cfg
from model import RPN3D
from utils import *
from utils.kitti_loader import iterate_data, sample_test_data
from train_hook import check_if_should_pause




parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=160,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
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
