import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import torch
import torch_tensorrt
from tqdm import tqdm

class NativeTorchBenchmark():
    
    def __init__(self,
                 n_infers,
                 batch_size,
                 samples,
                 model_arch,
                 model_ckpt,
                 device='cuda',
                 warmup=True):
        '''
        n_infers: no. of inference times
        '''
        self.n_infers = n_infers
        self.batch_size = batch_size
        self.samples = torch.Tensor(samples)
        self.device = device
        self.warmup = warmup
        self.model = model_arch.to(self.device)
        self.model.load_state_dict(torch.load(model_ckpt))
        
    def benchmark(self):
        self.model.eval()
        with torch.no_grad():
            if self.warmup:
                print('Warmup...')
                for i in range(5):
                    self.model(self.samples.to(self.device)).cpu()
            print('Start benchmarking')
            t1 = time.time()
            for i in tqdm(range(self.n_infers)):
                pred = self.model(self.samples.to(self.device)).cpu()
        t2 = time.time()
        throughputs = self.batch_size*self.n_infers/(t2-t1)
        latency = (t2-t1)/self.n_infers*1000
        print('Throughputs:', round(throughputs, 4), 'imgs/sec')
        print('Latency:', round(latency, 4), 'ms')
        

class TorchScriptBenchmark(NativeTorchBenchmark):
    
    def __init__(self,
                 n_infers,
                 batch_size,
                 samples,
                 model_arch,
                 model_ckpt,
                 device='cuda',
                 warmup=True):
        '''
        n_infers: no. of inference times
        '''
        self.n_infers = n_infers
        self.batch_size = batch_size
        self.samples = torch.Tensor(samples)
        self.device = device
        self.warmup = warmup
        self.model = model_arch
        self.model.load_state_dict(torch.load(model_ckpt))
        self.model = torch.jit.script(self.model).to(self.device)

class TorchTensorRTBenchmark(NativeTorchBenchmark):
    
    def __init__(self,
                 n_infers,
                 batch_size,
                 samples,
                 model_path,
                 device='cuda',
                 warmup=True):
        '''
        n_infers: no. of inference times
        '''
        self.n_infers = n_infers
        self.batch_size = batch_size
        self.samples = torch.Tensor(samples).half()
        self.device = device
        self.warmup = warmup
        self.model = torch.jit.load(model_path)

class TensorRTBehcnmark():
    
    def __init__(self,
                 n_infers,
                 batch_size,
                 samples,
                 engine_path,
                 warmup=True):
        
        self.engine_path = engine_path
        self.n_infers = n_infers
        self.batch_size = batch_size
        self.samples = samples
        self.warmup = warmup
        
    def TRT_setup(self, engine_path):
        TRT_LOGGER = trt.Logger()
        trt.init_libnvinfer_plugins(None,'')
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(self.samples)
                input_memory = cuda.mem_alloc(self.samples.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
        return context, input_buffer, input_memory, output_buffer, output_memory, bindings

    def infer(self,
              context,
              input_buffer,
              input_memory,
              output_buffer,
              output_memory,
              bindings):
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()

        return output_buffer
    
    def benchmark(self):
        context, input_buffer, input_memory, output_buffer, output_memory, bindings = self.TRT_setup(self.engine_path)
        if self.warmup:
            print('Warmup...')
            for i in range(5):
                pred = self.infer(context, input_buffer, input_memory, output_buffer, output_memory, bindings)
        print('Start benchmarking')
        t1 = time.time()
        for i in tqdm(range(self.n_infers)):
            pred = self.infer(context, input_buffer, input_memory, output_buffer, output_memory, bindings)
        t2 = time.time()
        throughputs = self.batch_size*self.n_infers/(t2-t1)
        latency = (t2-t1)/self.n_infers*1000
        print('Throughputs:', round(throughputs, 4), 'imgs/sec')
        print('Latency:', round(latency, 4), 'ms')
        
