# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#####################################追加##################################
import sys
sys.path.append("/content/drive/MyDrive/00Colab Notebooks/07Datasets/deep-high-resolution-net.pytorch")
from verification.seed import seed_everything
from verification.gpu import Gpu_used
from verification.time_veri import Time_veri
from verification.make_log import setup_logger
seed_everything()
gpu_used = Gpu_used()
time_veri1 = Time_veri()
time_veri2 = Time_veri()
time_veri3 = Time_veri()
FILE_NAME="hr-time"
logger = setup_logger(FILE_NAME=FILE_NAME)
logger.info("log:{}".format("log_test"))
logger.info("gpu_start{}".format(gpu_used.gpuinfo()))
#####################################追加##################################


import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
#######################################
from core.function_hr import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    ###################コメントアウト
    # logger, final_output_dir, tb_log_dir = create_logger(
    #     cfg, args.cfg, 'valid')
    final_output_dir="/content/drive/MyDrive/00Colab Notebooks/07Datasets/deep-high-resolution-net.pytorch/verification/_output_dir"
    tb_log_dir="/content/drive/MyDrive/00Colab Notebooks/07Datasets/deep-high-resolution-net.pytorch/verification/_output_dir"

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    #lib.models.pose_hrenet.get_pose_net()
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        logger.info('=> test.batch.size {}'.format(cfg.TEST.BATCH_SIZE_PER_GPU))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # model = model.to(device)
    # model.half()
    logger.info("gpu_model:{}".format(gpu_used.gpuinfo()))

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
###################################################################
#eval()()→dataset.Dataset(引数)としてデータセットを読んでいる
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
