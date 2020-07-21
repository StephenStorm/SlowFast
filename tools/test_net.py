#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter

# stephen add for visualization
import sys
import cv2
import pandas as pd
from time import time
import os
# steohen add end

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # stephen add ,prepare label to get class name
    label_path = cfg.DATA.PATH_TO_LABEL
    labels_df = pd.read_csv(label_path)
    tlabels = labels_df['name'].values

    
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    start = time()
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        
        # Transfer the data to the current GPU device.
        labels = labels.cuda()
        video_idx = video_idx.cuda()
        '''
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)
        '''

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach().cpu(),
                ori_boxes.detach().cpu(),
                metadata.detach().cpu(),
            )
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds = model(inputs)
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather(
                    [preds, labels, video_idx]
                )

            test_meter.iter_toc()
            # Update and log stats.
            # here origin method don't return ,but stephen add 
            tres = test_meter.update_stats(
                preds.detach().cpu(),
                labels.detach().cpu(),
                video_idx.detach().cpu(),
            )
            test_meter.log_iter_stats(cur_iter)

            # stephen add for visualization
            
            if tres is not None:
                dur = time() - start
                pred_idx = tres
                label_idx = labels[0]
                
                cap = cv2.VideoCapture(meta['path'][0])
                if pred_idx != label_idx:
                    video_writer_path = os.path.join('./test/checkpoint_300/20200721_park_9clips/fail', os.path.basename(meta['path'][0]))
                else:
                    video_writer_path = os.path.join('./test/checkpoint_300/20200721_park_9clips/success', os.path.basename(meta['path'][0]))
                video_writer = cv2.VideoWriter(video_writer_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (int(cap.get(3)), int(cap.get(4))))
                # cv2.namedWindow('wid')
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        cv2.putText(frame, 'label: {}'.format(tlabels[label_idx]), (20, 20), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=(0, 0, 255), thickness=2)
                        cv2.putText(frame, 'pred: {}'.format(tlabels[pred_idx]), (20, 40), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=(0, 0, 255), thickness=2)
                        cv2.putText(frame, 'time: {:.3f}'.format(dur), (20, 70), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=(255, 0, 255), thickness=2)
                        # cv2.imshow('wid', frame)
                        
                        video_writer.write(frame)
                        cv2.waitKey(1)
                        #     break
                    else:
                        
                        # if cv2.waitKey() == 27:
                        #     sys.exit()
                        break
                video_writer.release()
                cap.release()
                cv2.destroyAllWindows()


                start = time()
            
            # stephen add end
            

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    # stephen add
    if cfg.TRAIN.TOP5:
        tks = (1, 5)
    else:
        tks = (1,)
    test_meter.finalize_metrics(tks)
    test_meter.reset()


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, is_train=False)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg)
