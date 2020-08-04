import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys, os
TRT_LOGGER = trt.Logger()

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


def build_my_engine(engine_file_path, onnx_file_path):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    # if os.path.exists(engine_file_path):
    if False:
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30
    #         The maximum GPU temporary memory which the engine can use at execution time.
            builder.fp16_mode = True
            builder.max_batch_size = 3
            config = builder.create_builder_config()
            profile = builder.create_optimization_profile()
            # set_shape(self: tensorrt.tensorrt.IOptimizationProfile, input: str, 
            # min: tensorrt.tensorrt.Dims, opt: tensorrt.tensorrt.Dims, 
            # max: tensorrt.tensorrt.Dims) → None
            profile.set_shape("slow", (1, 3, 8, 256, 256), (1, 3, 8, 256, 256), (2, 3, 8, 256, 256))
            profile.set_shape("fast", (1, 3, 32, 256, 256), (1, 3, 32, 256, 256), (2, 3, 32, 256, 256))
            config.add_optimization_profile(profile)
            # This function must be called at least once if the network has dynamic or shape input tensors.
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run export_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('error occurd ~')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            # engine = builder.build_cuda_engine(network)
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            # with open(engine_file_path, "wb") as f:
            #     f.write(engine.serialize())
            print(profile.get_shape('slow'))
            print(profile.get_shape('fast'))
            print(profile.get_shape_input('slow'))
            return engine

'''
    
context.set_binding_shape(0, (3, 150, 250))

profile = builder.create_optimization_profile();
profile.set_shape("foo", (3, 100, 200), (3, 150, 250), (3, 200, 300)) 
config.add_optimization_profile(profile)

with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
    config.max_workspace_size = 1 << 20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    with builder.build_engine(network, config) as engine:

# Do inference here.
'''
onnx_file_path = '/home/stephen/workspace/ActionRecognition/my_SlowFast/onnx/slowfast_mul_batch_sim.onnx'
onnx_file_path2 = '/home/stephen/workspace/ActionRecognition/onnx_trt/test15_sim.onnx'
engine_file_path = '/home/stephen/workspace/ActionRecognition/my_SlowFast/onnx/slowfast_mul_batch.trt'
# engine_file_path = ''
'''

# engine = get_engine(onnx_file_path)
if engine is None:
    print('fail build engine')
print(engine.get_binding_shape(0),
      engine.get_binding_shape(1),
      engine.get_binding_shape(2)
     )
# The number of binding indices.
print('num_bindings: {}'.format(engine.num_bindings))
# The maximum batch size which can be used for inference. implicit 1
print('max batch size: {}'.format(engine.max_batch_size))
# 优化合并后的层数
print('num_layers: {}'.format(engine.num_layers))
# Workspace will be allocated for each IExecutionContext
print('max_workspace_size: {}'.format(engine.max_workspace_size))
# num_optimization_profiles
print('optimizition profiles for this engine: {}'.format(engine.num_optimization_profiles))
'''
engine = build_my_engine(engine_file_path, onnx_file_path)
with  engine.create_execution_context() as context:
    print(context.get_binding_shape(0))
