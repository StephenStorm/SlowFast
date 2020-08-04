#!/usr/bin/env python2
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
# 上述语句未显式调用，但是必须引入，否则会报错
import cv2
# import onnx


import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], ".."))

import common
import random
# stephen add :
import torch
# torch.backends.cudnn.enabled = False
from slowfast.datasets import loader
from slowfast.config.defaults import get_cfg
import math
import matplotlib.pyplot as plt
from slowfast.datasets.build import build_dataset


TRT_LOGGER = trt.Logger()


# slow [1, 3, 8, 256, 256] 
# fast [1, 3, 32, 256, 256]

frames_fast = 32
alpha = 4

size = 256
channel = 3

class_dict = {0:'not_wave', 1:'wave'}




def get_inputs_from_video(video_path):
    mean = np.array([0.45, 0.45, 0.45])
    std = np.array([0.225, 0.225, 0.225])
    sampling_rate = 2
    alpha = 4
    sampling_frames_num = frames_fast * sampling_rate
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    if frames_num < sampling_frames_num:
        return None
    assert frames_num > sampling_frames_num, 'video length < {}!'.format(sampling_frames_num)
    # start = random.randrange(frames_num - sampling_frames_num)
    start = 0
    # print('start frame: {}'.format(start))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    count = 0

    width = cap.get(3)
    height = cap.get(4)
    # print('origin height:{}, origin width {}'.format(height, width))
    # 决定resize尺寸
    new_width = size
    new_height = size

    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    
    # print('resize height:{}, resize width {}'.format(new_height, new_width))
    
    # print('crop height origin:{}, crop width origin:{}'.format(y_offset, x_offset))
    total_frames = np.zeros((sampling_frames_num, new_height, new_width, channel))
    while cap.isOpened():
        res, frame = cap.read()
        if res and count < sampling_frames_num:
            tmp = cv2.resize(frame, (new_width, new_height))
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            tmp = tmp / 255.0
            tmp = tmp - mean
            tmp = tmp / std
            # plt.imshow(tmp)
            # plt.pause(0.0001)
            total_frames[count] = tmp
            count = count + 1
        else :
            break
    # t, w, h, c -> c, t, w, c
    total_frames = np.transpose(total_frames, [3, 0, 1, 2])
    # fast = fast[:, 0:64:2, y_offset : y_offset + size, x_offset : x_offset + size]
    fast = np.zeros((3, 3, frames_fast, size, size),dtype=np.float32)# numpy 默认精度是float64， 此处dtype不能省略，否则在推断时由于精度的问题，会出现结果错误或者nan的情况。
    slow = np.zeros((3, 3, frames_fast // alpha, size, size),dtype=np.float32)
    if new_height > new_width:
        # offset along width
        x_offset = [0] * 3
        # offset along height
        y_offset = [0, np.ceil((new_height - size) / 2), new_height - size]
    else:
        # width > height
        x_offset = [0, int(np.ceil((new_width - size) / 2)), new_width - size]
        y_offset = [0] * 3
    # print('offset along width: {}'.format(x_offset))
    # print('offset along height: {}'.format(y_offset))
    for i in range(3):
        '''
        # 高度裁剪起始点初始值
        y_offset = int(math.ceil((new_height - size) / 2))
        # 宽度裁剪起始点初始值
        x_offset = int(math.ceil((new_width - size) / 2))

        if new_height > new_width:
            if i == 0:
                y_offset = 0
            elif i == 2:
                y_offset = new_height - size
        else:
            if i == 0:
                x_offset = 0
            elif i == 2:
                x_offset = new_width - size
        '''
        # # c, t, h, w
        fastt = np.array(total_frames[:, 0:64:sampling_rate, y_offset[i]:y_offset[i]+size, x_offset[i]:x_offset[i]+size])
        slowt = np.array(total_frames[:, 0:64:sampling_rate * alpha, y_offset[i]:y_offset[i]+size, x_offset[i]:x_offset[i]+size])
        fast[i] = np.ascontiguousarray(fastt)
        slow[i] = np.ascontiguousarray(slowt)
        
    print('final slow:{},final fast:{}'.format(slow.shape, fast.shape))
    return [slow, fast]


def get_inputs_from_video_by_torch(video_path):
    mean = np.array([0.45, 0.45, 0.45])
    std = np.array([0.225, 0.225, 0.225])
    sampling_rate = 2
    alpha = 4
    sampling_frames_num = frames_fast * sampling_rate
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    if frames_num < sampling_frames_num:
        return None
    assert frames_num > sampling_frames_num, 'video length < {}!'.format(sampling_frames_num)
    # start = random.randrange(frames_num - sampling_frames_num)
    start = 0
    # print('start frame: {}'.format(start))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    count = 0

    width = cap.get(3)
    height = cap.get(4)
    # print('origin height:{}, origin width {}'.format(height, width))
    # 决定resize尺寸
    new_width = size
    new_height = size

    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    
    # print('resize height:{}, resize width {}'.format(new_height, new_width))
    
    # print('crop height origin:{}, crop width origin:{}'.format(y_offset, x_offset))
    total_frames = np.zeros((sampling_frames_num, new_height, new_width, channel))
    while cap.isOpened():
        res, frame = cap.read()
        if res and count < sampling_frames_num:
            tmp = cv2.resize(frame, (new_width, new_height))
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            tmp = tmp / 255.0
            tmp = tmp - mean
            tmp = tmp / std
            # plt.imshow(tmp)
            # plt.pause(0.0001)
            total_frames[count] = tmp
            count = count + 1
        else :
            break
    # t, w, h, c -> c, t, w, c
    total_frames = np.transpose(total_frames, [3, 0, 1, 2])
    # fast = fast[:, 0:64:2, y_offset : y_offset + size, x_offset : x_offset + size]\
    # fast = np.zeros((3, 3, frames_fast, size, size))
    # slow = np.zeros((3, 3, frames_fast // alpha, size, size))
    fast = torch.zeros((3, 3, frames_fast, size, size))
    slow = torch.zeros((3, 3, frames_fast // alpha, size, size))
    if new_height > new_width:
        # offset along width
        x_offset = [0] * 3
        # offset along height
        y_offset = [0, np.ceil((new_height - size) / 2), new_height - size]
    else:
        # width > height
        x_offset = [0, int(np.ceil((new_width - size) / 2)), new_width - size]
        y_offset = [0] * 3
    # print('offset along width: {}'.format(x_offset))
    # print('offset along height: {}'.format(y_offset))
    for i in range(3):
        '''
        # 高度裁剪起始点初始值
        y_offset = int(math.ceil((new_height - size) / 2))
        # 宽度裁剪起始点初始值
        x_offset = int(math.ceil((new_width - size) / 2))

        if new_height > new_width:
            if i == 0:
                y_offset = 0
            elif i == 2:
                y_offset = new_height - size
        else:
            if i == 0:
                x_offset = 0
            elif i == 2:
                x_offset = new_width - size
        '''
        # # c, t, h, w
        fast[i] = torch.from_numpy(total_frames[:, 0:64:sampling_rate, y_offset[i]:y_offset[i]+size, x_offset[i]:x_offset[i]+size]).contiguous()
        slow[i] = torch.from_numpy(total_frames[:, 0:64:sampling_rate * alpha, y_offset[i]:y_offset[i]+size, x_offset[i]:x_offset[i]+size]).contiguous()
        
    print('final slow:{},final fast:{}'.format(slow.shape, fast.shape))
    # print(np.max(slow))
    return [slow.numpy(), fast.numpy()]

def test_input():
    # test_path = '/home/stephen/workspace/Data/wave_stop/resized_clips/wave_resized/clips59179.mp4'
    test_path = '/home/stephen/workspace/Data/my_wave.mp4'
    input = get_inputs_from_video(test_path)
    # return 0
    print(input[0].shape)# 3, 3, 8, 256, 256
    print(input[1].shape)# 3, 3, 32, 256, 256  in (b, c, t, h, w)
    for i in range(32):
        img = input[1][0, :, i, :, :].copy()
        img = np.transpose(img, [1,2,0])
        # plt.figure()
        plt.imshow(img)
        plt.pause(0.001)

def test_input_torch():
    cfg_file = '/home/stephen/workspace/ActionRecognition/my_SlowFast/configs/myconfig/Kinetics/SLOWFAST_8x8_R50_test_local.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    test_loader = loader.construct_loader(cfg, "test")
    count = 0
    for cur_iter, (clips, labels, video_idx, meta) in enumerate(test_loader):
        fast = clips[0][0,:, :, :, :]
        imgs = fast.permute((1, 2, 3, 0))
        print(imgs.shape[0])
        for i in range(imgs.shape[0]):
            img = imgs[i]
            plt.imshow(img)
            # plt.show()
            plt.pause(0.001)
            # plt.show()
        # a = input('tt')
        # count = count + 1
        # if count > 5:
        #     break
        



def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run export_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))
            
            # print(type(network))
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            # with open(engine_file_path, "wb") as f:
            #     f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main(method = 'max'):
    assert method in ['max','sum']
    """Create a TensorRT engine for ONNX-based slowfast and run inference."""
    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'test_sim.onnx'
    engine_file_path = "onnx/slowfast_sim.trt"
    input_video_path = '/home/stephen/workspace/Data/wave_stop/resized_clips/wave_resized/clips59179.mp4'
    # Download a dog image and save it to the following file path:
    # input = get_inputs_from_video(input_video_path)
    '''
    cfg_file = '/home/stephen/workspace/ActionRecognition/my_SlowFast/configs/myconfig/Kinetics/SLOWFAST_8x8_R50_test_local.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    test_loader = loader.construct_loader(cfg, "test")
    '''

    # Do inference with TensorRT
    # trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # print('Running inference on video {}...'.format(input_video_path))
        print('inputs dim:{}, outputs dim: {}'.format(len(inputs), len(outputs)))
        correct = 0
        cc = 0
        with open('data_process/data/local/test.csv', 'r') as test, open('fail.txt', 'w') as fail:
            lines = test.readlines()
            total = len(lines)
            res = np.zeros((total, 2))
            for line in lines:
                (path, label) = line[:-1].split(',')
                tinput = get_inputs_from_video(path)
                if tinput is None:
                    total = total - 1
                    continue
                for i in range(3):
                    inputs[0].host = tinput[0][i]
                    inputs[1].host = tinput[1][i]
                    (trt_outputs, _) = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                    print(trt_outputs[0][:2])
                    if method == 'sum':
                        res[cc] = res[cc] + trt_outputs[0][:2].reshape(1,-1)
                    else:
                        res[cc] = np.maximum(res[cc], trt_outputs[0][:2].reshape(1,-1))
                    # print('res: {}'.format(res))
                index = int(np.argmax(res[cc]))
                label = int(label)
                if index == label:
                    correct = correct + 1
                else:
                    # fail.write('{}\t{}\n'.format(cc + 1, path))
                    print(path)
                    
                    print(''.center(60, '-'))
                # print(path)
                print('res: {}'.format(res[cc]))
                print('predict res: {}, ground_truth: {}'.format(class_dict[index], class_dict[label]))
                cc = cc + 1
                # break
                
            print('accuracy: {:6.3f}'.format(correct / total * 100))

def infer_with_torch():
    """Create a TensorRT engine for ONNX-based slowfast and run inference."""
    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'test_sim.onnx'
    engine_file_path = "slowfast_sim.trt"
    # input_video_path = '/home/stephen/workspace/Data/wave_stop/resized_clips/wave_resized/clips59179.mp4'

    cfg_file = '/home/stephen/workspace/ActionRecognition/my_SlowFast/configs/myconfig/Kinetics/SLOWFAST_8x8_R50_test_local.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    test_loader = loader.construct_loader(cfg, "test")

    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('inputs dim:{}, outputs dim: {}'.format(len(inputs), len(outputs)))
        tt = 0
        for cur_iter, (clips, labels, video_idx, meta) in enumerate(test_loader):
            
            # print('slow: {}, fast: {}'.format(clips[0].shape, clips[1].shape))
            # print(clips[0].is_contiguous(), clips[1].is_contiguous()) true , true
            # print(clips[0].device, clips[1].device)
            slow = clips[0].numpy()
            fast = clips[1].numpy()
            # print(type(slow), type(fast))  np.array
            # print('slow: {}, fast: {}'.format(slow.shape, fast.shape))
            # '''
            inputs[0].host = slow
            inputs[1].host = fast
            print('slow size:{}, fast size:{}'.format(inputs[0].host.shape, inputs[1].host.shape))
            print('begin inference:\n')
            if tt > 5:
                break
            tt = tt + 1
            # '''
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print('end inference')
            print('cur_iter: {}'.format(cur_iter))
            # '''
            # print(len(trt_outputs)) # 1
            # print(type(trt_outputs[0])) # np.ndarray
            # print(trt_outputs[0].shape) # (64, )
            print(trt_outputs[0])
            # '''
        '''
        correct = 0
        cc = 0
        with open('data_process/data/local/test.csv', 'r') as test, open('fail.txt', 'w') as fail:
            lines = test.readlines()
            total = len(lines)
            res = np.zeros((total, 2))
            for line in lines:
                (path, label) = line[:-1].split(',')

                tinput = get_inputs_from_video(path)
                if tinput is None:
                    total = total - 1
                    continue
                for i in range(3):

                    
                    inputs[0].host = tinput[0][i]
                    inputs[1].host = tinput[1][i]
                    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                    print(trt_outputs[0][:2])
                    res[cc] = np.maximum(res[cc], trt_outputs[0][:2].reshape(1,-1))
                    
                    # print('res: {}'.format(res))
                index = int(np.argmax(res[cc]))
                label = int(label)
                if index == label:
                    correct = correct + 1
                else:
                    fail.write('{}\t{}\n'.format(cc + 1, path))
                print('res: {}'.format(res[cc]))
                print('predict res: {}, ground_truth: {}'.format(class_dict[index], class_dict[label]))
                cc = cc + 1
            print('accuracy: {:6.3f}'.format(correct / total * 100))
        '''
    
def infer_with_dataset():

    """Create a TensorRT engine for ONNX-based slowfast and run inference."""
    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'test_sim.onnx'
    engine_file_path = "/home/stephen/workspace/ActionRecognition/onnx_trt/from_mul_batch_sim_with_trtexec.trt"
    # input_video_path = '/home/stephen/workspace/Data/wave_stop/resized_clips/wave_resized/clips59179.mp4'

    cfg_file = '/home/stephen/workspace/ActionRecognition/my_SlowFast/configs/myconfig/Kinetics/SLOWFAST_8x8_R50_test_local.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    
    seg_nums = cfg.TEST.NUM_ENSEMBLE_VIEWS
    crop_nums = cfg.TEST.NUM_SPATIAL_CROPS

    clips_nums = seg_nums * crop_nums

    dataset = build_dataset('kinetics', cfg, 'test')
    dataset_len = len(dataset)
    video_nums = dataset_len / clips_nums
    right_total = video_nums
    print('vid_num {}'.format(video_nums))
    res = np.zeros((int(video_nums),2))
    # Do inference with TensorRT
    total_time = 0
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        for i in range(dataset_len):
            vid_id = i //  clips_nums

            # data返回值形式 ： frames(tensor), label(int), index(int), {}
            frames = dataset[i][0]
            label = dataset[i][1]
            slow = frames[0].unsqueeze(0).contiguous().numpy()
            fast = frames[1].unsqueeze(0).contiguous().numpy()
            inputs[0].host = slow
            inputs[1].host = fast
            # print('slow size:{}, fast size:{}'.format(inputs[0].host.shape, inputs[1].host.shape))
            (trt_outputs, dur) = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            total_time = total_time + dur
            print('{}'.format(trt_outputs[0][:2]))
            if cfg.DATA.ENSEMBLE_METHOD == 'max':
                res[vid_id] = np.maximum(res[vid_id], trt_outputs[0][:2].reshape(1, -1))
            else:
                res[vid_id] = res[vid_id] + trt_outputs[0][:2].reshape(1, -1)
            if (i+1) % clips_nums == 0:
                pre_ind = np.argmax(res[vid_id])
                print('label: {}, preds: {}({:4.3f})'.format(class_dict[label], class_dict[pre_ind], res[vid_id][pre_ind]))
                if label != pre_ind:
                    print(''.center(60, '-'))
                    right_total = right_total - 1
        print('accuracy: {:6.4f}'.format(right_total *100 / video_nums))
        print('avg time / video: {:4.3f}s, avg time / clips: {:6.2f} ms'.format(total_time / video_nums, total_time * 1000 / dataset_len))


def test():
    input_video_path = '/home/stephen/workspace/Data/wave_stop/resized_clips/negative_resized/clips61046.mp4'
    from_torch = get_inputs_from_video_by_torch(input_video_path)
    from_np = get_inputs_from_video(input_video_path)
    
    # print(from_torch[0].shape, from_torch[1].shape)
    # print(from_np[0].shape, from_np[1].shape)
    # print(np.max(from_torch[1]), np.max(from_np[1]))
    # dif = from_torch[1] - from_np[1]
    # print(from_torch[1] == from_np[1])
    # c, t, size, size
    print(from_torch[1].dtype,from_np[1].dtype)
    for i in range(10):
        index = random.randint(0, 256)
        print(from_torch[1][0,0,0,0,i],from_np[1][0,0,0,0,i])
        # print(from_torch[1].dtype,from_np[1].dtype)
    # print(from_torch[1][0,0,0,0,245],from_np[1][0,0,0,0,245])
    # print(np.count_nonzero(from_np[1]!=from_np[1]))
    return 1
    onnx_file_path = 'test_sim.onnx'
    engine_file_path = "onnx/slowfast_sim.trt"
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # print('Running inference on video {}...'.format(input_video_path))
        tinput = get_inputs_from_video(input_video_path)
        if tinput is None:
            total = total - 1
        for i in range(3):
            inputs[0].host = tinput[0][i]
            inputs[1].host = tinput[1][i]
            # print('inputs[0] shape: {}'.format(inputs[0].host.shape))
            (trt_outputs, _) = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print(trt_outputs[0][:2])


if __name__ == '__main__':

    # test_input()
    # print(''.center(90, '-'))
    # test_input_torch()
    # main('sum')
    test()


    # inputs = get_inputs_with_pytorch(0)
    # infer_with_torch()
    # infer_with_dataset()

    
                
    
    
    