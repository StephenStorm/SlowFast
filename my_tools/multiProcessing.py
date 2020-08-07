import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
# 上述语句未显式调用，但是必须引入，否则会报错
import cv2


import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], ".."))

import common
import random
# stephen add :
import math
import time

import threading
from threading import Thread

TRT_LOGGER = trt.Logger()

# wid_name = 'test'
# cv2.namedWindow(wid_name)


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



class Demo():
    def __init__(self, frame_fast = 32, sample_rate = 2):
        self.frame_fast = frame_fast
        self.sample_rate = sample_rate
        # sample frames nums
        self.frame_nums = self.frame_fast * self.sample_rate
        self.alpha = 4
        self.cam_height = 720
        self.cam_width = 1280
        # self.frames = np.zeros((frame_fast * sample_rate, self.cam_height, self.cam_width), dtype = 'uint8')

        self.frames_lock = threading.Lock()
        self.clips_lock = threading.Lock()
        # first resize , Consistent with training
        self.resize_width = 640
        self.resize_height = 480
        # crop size
        self.size = 256
        self.current_clip = None
        self.exit_flag = False
        self.class_dict = {0:'not_wave', 1:'wave'}


        self.cap = cv2.VideoCapture(0)
        self.cap.set(4, self.cam_height)  # height
        self.cap.set(3, self.cam_width * 2)  # width


        self.frames = self.get_input_from_cam()
        self.current_clip = self.frames.copy()
        

    # return frames of size [frame_nums, heitht, width, 3(channel)]
    def get_input_from_cam(self):
        # wid_name = 'test'
        frames = np.zeros((self.frame_nums, self.cam_height, self.cam_width, 3), dtype = 'uint8')
        count = 0
        while self.cap.isOpened():
            res, frame = self.cap.read()
            if res and count < self.frame_nums:
                # 取出做相机画面
                frame = frame[:self.cam_height, :self.cam_width, :]
                # frame = cv2.resize(frame, (256, 256))
                # print(frame.shape)
                # cv2.imshow('tt', frame)
                # cv2.waitKey(20)
                frames[count] = frame
                count += 1
            else :
                break
        return frames


    def display_clips(self):
        cv2.namedWindow('iner')
        count = 0
        while not self.exit_flag and count < 300:
            # print(self.current_clip.shape)
            len = self.current_clip.shape[0]
            # print(threading.currentThread())
            # print('len: {}'.format(len))
            
            for i in range(len):
                frame = self.current_clip[i]
                # print(frame.shape)
                cv2.imshow('iner', frame)
                cv2.waitKey(30)
            count += 1

        print('display thread exit successfully')


    def process_frames(self):
        mean = np.array([0.45, 0.45, 0.45])
        std = np.array([0.225, 0.225, 0.225])
        alpha = 4

        width = self.resize_width
        height = self.resize_height
        # 决定resize尺寸
        new_width = self.size
        new_height = self.size
        if width < height:
            new_height = int(math.floor((float(height) / width) * self.size))
        else:
            new_width = int(math.floor((float(width) / height) * self.size))
        count = 0
        # print('resize height:{}, resize width {}'.format(new_height, new_width))
        # print('crop height origin:{}, crop width origin:{}'.format(y_offset, x_offset))
        total_frames = np.zeros((self.frame_fast, new_height, new_width, 3))
        for i in range(self.frame_nums):
            if i % self.sample_rate == 0:
                frame = self.frames[i]
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                tmp = cv2.resize(frame, (new_width, new_height))
                tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
                tmp = tmp / 255.0
                tmp = tmp - mean
                tmp = tmp / std
                total_frames[count] = tmp
                count += 1
        # t, w, h, c -> c, t, w, c
        total_frames = np.transpose(total_frames, [3, 0, 1, 2])
        fast = np.zeros((3, 3, self.frame_fast, self.size, self.size), dtype=np.float32)
        # numpy 默认精度是float64， 此处dtype不能省略，否则在推断时由于精度的问题，会出现结果错误或者nan的情况。
        slow = np.zeros((3, 3, self.frame_fast // alpha, self.size, self.size), dtype=np.float32)
        if new_height > new_width:
            # offset along width
            x_offset = [0] * 3
            # offset along height
            y_offset = [0, np.ceil((new_height - self.size) / 2), new_height - self.size]
        else:
            # width > height
            x_offset = [0, int(np.ceil((new_width - self.size) / 2)), new_width - self.size]
            y_offset = [0] * 3
        for i in range(3):
            # # c, t, h, w
            fastt = np.array(total_frames[:, :, y_offset[i]:y_offset[i] + self.size, x_offset[i]:x_offset[i] + self.size])
            slowt = np.array(
                total_frames[:, 0:self.frame_fast:alpha, y_offset[i]:y_offset[i] + self.size, x_offset[i]:x_offset[i] + self.size])
            fast[i] = np.ascontiguousarray(fastt)
            slow[i] = np.ascontiguousarray(slowt)

        print('final slow:{},final fast:{}'.format(slow.shape, fast.shape))
        return [slow, fast]

    def update(self):
        # with self.frames_lock:
        with self.clips_lock:
            self.frames = self.get_input_from_cam()
            self.current_clip = self.frames





    def infer_with_trt(self, method = 'max'):
        onnx_file_path = 'onnx/test_sim.onnx'
        engine_file_path = "onnx/slowfast_sim.trt"

        th1 = Thread(target = self.display_clips)
        th1.start()
        # th1.join()
        with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            res = np.zeros((1, 2))
            while True:
                # print('process...')
                # print(threading.currentThread())
                (slow, fast) = self.process_frames()
                res.fill(0)
                for i in range(3):
                    inputs[0].host = slow[i]
                    inputs[1].host = fast[i]
                    (trt_outputs, _) = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                                           stream=stream)
                    print(trt_outputs[0][:2])
                    if method == 'sum':
                        res[0] = res[0] + trt_outputs[0][:2].reshape(1, -1)
                    else:
                        res[0] = np.maximum(res[0], trt_outputs[0][:2].reshape(1, -1))
                print('res: {}'.format(res[0]))
                index = int(np.argmax(res[0]))
                print('predict res: {}'.format(self.class_dict[index]))
                # print('sleep done')
                self.update()





if __name__ == '__main__':
    wid_name = 'test'

    demo1 = Demo()
    print(demo1.frames.shape)
    # print(np.min(demo1.frames))
    # demo1.display_clips()
    demo1.infer_with_trt()