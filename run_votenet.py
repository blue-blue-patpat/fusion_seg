# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import numpy as np
from datetime import datetime
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='mmDetection', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--output_dir', default='', type=str, help='path where to save')
parser.add_argument('--num_points', type=int, default=5000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--use_feature', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
parser.add_argument('--features', default=3, type=int, help='dim of features')
parser.add_argument('--test_data', default='lab1', type=str, help='type of test data, test, rain, smoke, night, occlusion, confusion')
parser.add_argument('--train', dest='train', action='store_true', help='train or test')
parser.add_argument('--data_path', default='/home/nesc525/drivers/1,/home/nesc525/drivers/2,/home/nesc525/drivers/3', type=str, help='dataset')
parser.add_argument('--device', default=0, type=str, help='cuda device')
ARGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.device

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from nn.votenet.models import votenet

from nn.votenet.pointnet2.pytorch_utils import BNMomentumScheduler
from nn.votenet.utils.tf_visualizer import Visualizer as TfVisualizer
from nn.votenet.models.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from nn.votenet.datasets.mm_detection import mmDetection, mmWaveDatasetConfig

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = ARGS.batch_size
NUM_POINT = ARGS.num_points
MAX_EPOCH = ARGS.max_epoch
BASE_LEARNING_RATE = ARGS.learning_rate
BN_DECAY_STEP = ARGS.bn_decay_step
BN_DECAY_RATE = ARGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in ARGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in ARGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
OUTPUT_DIR = ARGS.output_dir
LOG_DIR = os.path.join(OUTPUT_DIR, 'log')
DUMP_DIR = os.path.join(OUTPUT_DIR, 'dump')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.tar')

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and ARGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(ARGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
if ARGS.dataset == 'mmDetection':
    DATASET_CONFIG = mmWaveDatasetConfig()
    DATASET = mmDetection(
            driver_path=ARGS.data_path,
            features=ARGS.features,
            train=ARGS.train,
            test_data=ARGS.test_data,
    )

else:
    print('Unknown dataset %s. Exiting...'%(ARGS.dataset))
    exit(-1)

train_size = int(0.9 * len(DATASET))
eval_size = len(DATASET) - train_size
dataset_train, dataset_eval = torch.utils.data.random_split(DATASET, [train_size, eval_size])

if ARGS.train:
    TRAIN_DATALOADER = DataLoader(dataset_train, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=30, worker_init_fn=my_worker_init_fn)
    EVAL_DATALOADER = DataLoader(dataset_eval, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=30, worker_init_fn=my_worker_init_fn)
    print(len(TRAIN_DATALOADER), len(EVAL_DATALOADER))
else:
    EVAL_DATALOADER = DataLoader(DATASET, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=30, worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
MODEL = importlib.import_module(ARGS.model) # import network module
# torch.cuda.set_device(ARGS.device)
device = torch.device('cuda')

Detector = MODEL.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=ARGS.num_target,
               input_feature_dim=DATASET.features,
               vote_factor=ARGS.vote_factor,
               sampling=ARGS.cluster_sampling)

# if torch.cuda.device_count() > 1:
#     log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     net = nn.DataParallel(net)

net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=ARGS.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(ARGS, 'train')
TEST_VISUALIZER = TfVisualizer(ARGS, 'test')


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net(inputs)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
                (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    ap_calculator = APCalculator(ap_iou_thresh=ARGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type)
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(EVAL_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
                if batch_idx % 10 == 0:
                    log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if ARGS.dump_results and batch_idx == 0 and EPOCH_CNT %10 == 0:
            MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG) 

    # Log statistics
    if ARGS.train:
        TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
            (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        np.random.seed()
        train_one_epoch()
        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, CHECKPOINT_PATH)

def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    np.random.seed()
    loss = evaluate_one_epoch()

if __name__=='__main__':
    if ARGS.train:
        train(start_epoch)
    else:
        eval()
