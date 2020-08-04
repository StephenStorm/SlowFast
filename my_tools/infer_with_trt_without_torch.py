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
import math
import matplotlib.pyplot as plt


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
        # # c, t, h, w
        fastt = np.array(total_frames[:, 0:64:sampling_rate, y_offset[i]:y_offset[i]+size, x_offset[i]:x_offset[i]+size])
        slowt = np.array(total_frames[:, 0:64:sampling_rate * alpha, y_offset[i]:y_offset[i]+size, x_offset[i]:x_offset[i]+size])
        fast[i] = np.ascontiguousarray(fastt)
        slow[i] = np.ascontiguousarray(slowt)
        
    print('final slow:{},final fast:{}'.format(slow.shape, fast.shape))
    return [slow, fast]







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
    onnx_file_path = 'test_sim.onnx'
    engine_file_path = "onnx/slowfast_sim.trt"
    input_video_path = '/home/stephen/workspace/Data/wave_stop/resized_clips/wave_resized/clips59179.mp4'

    # Do inference with TensorRT
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
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
                print('res: {}'.format(res[cc]))
                index = int(np.argmax(res[cc]))
                label = int(label)
                if index == label:
                    correct = correct + 1
                else:
                    # fail.write('{}\t{}\n'.format(cc + 1, path))
                    print(path)
                    
                    print(''.center(60, '-'))
                
                print('predict res: {}, ground_truth: {}'.format(class_dict[index], class_dict[label]))
                cc = cc + 1
                
            print('accuracy: {:6.3f}'.format(correct / total * 100))




if __name__ == '__main__':

    main('sum')
