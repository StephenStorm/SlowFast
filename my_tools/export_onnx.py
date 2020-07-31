#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""
import torch
import numpy as np


import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
from slowfast.config.defaults import get_cfg

logger = logging.get_logger(__name__)


cfg_file = '/home/stephen/workspace/ActionRecognition/my_SlowFast/configs/myconfig/Kinetics/SLOWFAST_8x8_R50_test_local.yaml'

def export_onnx():
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
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
        print('load checkpoint from test checkpoint file')
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
    # test_loader = loader.construct_loader(cfg, "test")
    # logger.info("Testing model for {} iterations".format(len(test_loader)))
    model.eval()
    # model.to('cuda')
    model.cuda()
    # export model as onnx format
    batch_size = 1
    input1 = torch.rand(batch_size, 3, 8, 256, 256)
    input2 = torch.randn(batch_size, 3, 32, 256, 256)
    # input11 = input1.to('cuda')
    # input22 = input2.to('cuda')
    input11 = input1.cuda()
    input22 = input2.cuda()
    print(input11.device)
    input = [input11, input22]
    input_names = ['slow', 'fast']
    output_names = ['cls_res']
    export_onnx_file = "slowfast.onnx"					# 目的ONNX文件名
    torch.onnx.export(model,
                        input,
                        export_onnx_file,
                        verbose=True,
                        input_names = input_names,
                        output_names = output_names
    )
    
if __name__ == '__main__':
    export_onnx()




    
