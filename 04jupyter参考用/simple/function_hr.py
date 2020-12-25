# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


##################################################ここのなぞ
logger = logging.getLogger(__name__)
from verification.time_veri import Time_veri
from verification.gpu import Gpu_used
time_veri1 = Time_veri()
time_veri2 = Time_veri()
time_veri3 = Time_veri()
gpu_used = Gpu_used()


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):

    #########################コメントアウト             
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    time_veri1.time_start()
    with torch.no_grad():
        # end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            time_veri1.start_point(time_veri1.t_point0)
            time_veri2.start_point(time_veri2.t_point0)
            gpu_used.gpu_clear()
            
            # input = input.half()
            outputs = model(input)

            time_veri2.end_point(time_veri2.t_point0)
            time_veri2.start_point(time_veri2.t_point1)

            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            time_veri2.end_point(time_veri2.t_point1)
            time_veri2.start_point(time_veri2.t_point2)


            # if config.TEST.FLIP_TEST:
            # if config.TEST.FLIP_TEST:
            #     input_flipped = input.flip(3)
            #     outputs_flipped = model(input_flipped)

            #     if isinstance(outputs_flipped, list):
            #         output_flipped = outputs_flipped[-1]
            #     else:
            #         output_flipped = outputs_flipped

            #     output_flipped = flip_back(output_flipped.cpu().numpy(),
            #                                val_dataset.flip_pairs)
            #     output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


            #     # feature is not aligned, shift flipped heatmap for higher accuracy
            #     if config.TEST.SHIFT_HEATMAP:
            #         output_flipped[:, :, :, 1:] = \
            #             output_flipped.clone()[:, :, :, 0:-1]

            #     output = (output + output_flipped) * 0.5


            time_veri2.end_point(time_veri2.t_point2)
            time_veri2.start_point(time_veri2.t_point3)

            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.cuda(non_blocking=True)
            # loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            # losses.update(loss.item(), num_images)
            # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                            #  target.cpu().numpy())

            # acc.update(avg_acc, cnt)

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()


            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            time_veri2.end_point(time_veri2.t_point3)
            time_veri2.start_point(time_veri2.t_point4)

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            time_veri2.end_point(time_veri2.t_point4)
            time_veri2.start_point(time_veri2.t_point5)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            # if i % config.PRINT_FREQ == 0:
            #     # msg = 'Test: [{0}/{1}]\t' \
            #     #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
            #     #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
            #     #       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
            #     #           i, len(val_loader), batch_time=batch_time,
            #     #           loss=losses, acc=acc)
                
            #     msg = 'Test: [{0}/{1}]\t'.format(i, len(val_loader))
            #     logger.info(msg)

            #     prefix = '{}_{}'.format(
            #         os.path.join(output_dir, 'val'), i)

                #######################################コメントアウト
                # save_debug_images(config, input, meta, target, pred*4, output, prefix)
            
            time_veri2.end_point(time_veri2.t_point5)
            time_veri1.end_point(time_veri1.t_point0)
            logger.info("gpu_batch1:{}".format(gpu_used.gpuinfo()))
            gpu_used.gpu_clear()
            logger.info("gpu_batch2:{}".format(gpu_used.gpuinfo()))

        ########################################################################
        # print("評価の計算('◇')ゞ？")

        # name_values, perf_indicator = val_dataset.evaluate(
        #     config, all_preds, output_dir, all_boxes, image_path,
        #     filenames, imgnums
        # )

        # model_name = config.MODEL.NAME
        # if isinstance(name_values, list):
        #     for name_value in name_values:
        #         _print_name_value(name_value, model_name)
        # else:
        #     _print_name_value(name_values, model_name)

        # if writer_dict:
        #     writer = writer_dict['writer']
        #     global_steps = writer_dict['valid_global_steps']
        #     writer.add_scalar(
        #         'valid_loss',
        #         losses.avg,
        #         global_steps
        #     )
        #     writer.add_scalar(
        #         'valid_acc',
        #         acc.avg,
        #         global_steps
        #     )
        #     if isinstance(name_values, list):
        #         for name_value in name_values:
        #             writer.add_scalars(
        #                 'valid',
        #                 dict(name_value),
        #                 global_steps
        #             )
        #     else:
        #         writer.add_scalars(
        #             'valid',
        #             dict(name_values),
        #             global_steps
        #         )
        #     writer_dict['valid_global_steps'] = global_steps + 1

    #####################################追加##################################
    time_veri1.time_end()
    # logger.info(config)
    logger.info("log:{}".format("HR-NET処理中断終了"))

    
    logger.info("for文全体:{}".format(time_veri1.t_end - time_veri1.t_start))
    temp = (time_veri1.t_end - time_veri1.t_start)
    logger.info("for文の中:{}".format(time_veri1.time_sum_list(return_all=True)))
    # logger.info("ミニバッチのロード時間:{}".format(temp - time_veri2.all))
    logger.info("⓪～⑤の処理時間:{}sum：{}".format(time_veri2.time_sum_list(), time_veri2.all))
   
    logger.info("for文全体対fps_all:{}".format(1/(temp/512)))
    logger.info("for文の中対fps_in:{}".format(1/(time_veri1.all/512)))


    logger.info("gpu_propety:{}".format(gpu_used.gpu_property(device=0, torch_version=True)))
    logger.info("gpu_exit:{}".format(gpu_used.gpuinfo()))
    logger.info("gpu_max,list:{}{}".format(gpu_used.used_max, gpu_used.used_list))
    
    logger.info("コピペ{}".format([temp, time_veri1.all, time_veri2.time_sum_list()[0], time_veri2.time_sum_list()[1]
    , time_veri2.time_sum_list()[2], time_veri2.time_sum_list()[3], time_veri2.time_sum_list()[4], time_veri2.time_sum_list()[5] ]))
    import sys
    sys.exit()
    #####################################追加##################################

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
