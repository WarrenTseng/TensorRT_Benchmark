# TensorRT Benchmark and Triton Inference Server
"<a href="https://developer.nvidia.com/tensorrt">**NVIDIA® TensorRT™**</a>, an SDK for high-performance deep learning inference, includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for inference applications."  </br>
"<a href="https://developer.nvidia.com/nvidia-triton-inference-server">**Triton Inference Server**</a> is an open source inference serving software that streamlines AI inferencing. Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more. Triton supports inference across cloud, data center,edge and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia. Triton delivers optimized performance for many query types, including real time, batched, ensembles and audio/video streaming." </br>


## TensorRT Benchmark
These experimets are based on <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch">nvcr.io/nvidia/pytorch:23.04-py3</a> docker image. </br>

Please follow the instructions below to start the testing environment:
1. Pull the docker image and run
```
docker pull nvcr.io/nvidia/pytorch:23.04-py3
docker run -it --rm --shm-size=2g -p 8888:8888 --gpus='"device=0"' nvcr.io/nvidia/pytorch:23.04-py3
```

2. Clone this repo and install the requirements
```
git clone https://github.com/WarrenTseng/TensorRT_Benchmark
cd TensorRT_Benchmark/tensorrt_benchmark
pip install -r requirements.txt
```
 
3. Start jupyter lab
```
jupyter lab --ip 0.0.0.0 --allow-root
```

4. Benchmarking:
  - <a href="https://github.com/WarrenTseng/TensorRT_Benchmark/blob/main/tensorrt_benchmark/Classification.ipynb">Classification</a>
  - Detection
  - Segmentation

## TensorRT with Triton Inference Server
To compare TorchScript with TensorRT in Triton, we need the environments as below:
1. PyTorch: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch  
  - nvcr.io/nvidia/pytorch:23.04-py3
2. Triton Server and Client: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
  - Server: nvcr.io/nvidia/tritonserver:23.04-py3
  - Client: nvcr.io/nvidia/tritonserver:23.04-py3-sdk 

Please follow the instructions below to start the testing environment:
1. **Pull the docker images**:
```
docker pull nvcr.io/nvidia/pytorch:23.04-py3
docker pull nvcr.io/nvidia/tritonserver:23.04-py3
docker pull nvcr.io/nvidia/tritonserver:23.04-py3-sdk
```

2. **Preparing the models and Triton configs**:
  - Prepare the folders and clone this repo:
  ```
  TRITONPATH = /PATH/AS/TRITON/REPO
  SRCPATH = /PATH/AS/HOST/SRC
  mkdir $TRITONPATH
  mkdir $SRCPATH
  cd $SRCPATH
  git clone https://github.com/WarrenTseng/TensorRT_Benchmark
  cd TensorRT_Benchmark/tensorrt_with_triton/preparing
  mkdir models
  ```
  - Run the environment:
  ```
  docker run -it --rm --shm-size=2g -p 8888:8888 --gpus='"device=0"' -v $TRITONPATH:/repo -v $SRCPATH/TensorRT_Benchmark/tensorrt_with_triton/preparing:/ws -w /ws nvcr.io/nvidia/pytorch:23.04-py3
  ```
  - Preparing models
  - Preparing Triton configs
 
3. **Start Triton Inference Server**:
```
docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 --gpus='"device=0"' -v $TRITONPATH:/repo nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-store /repo
```
 
4. **Triton client and inference**:
  - Run the environment:
  ```
  docker run -it --rm -p 8889:8888 -v $SRCPATH/TensorRT_Benchmark/tensorrt_with_triton/client:/ws -w /ws nvcr.io/nvidia/tritonserver:23.04-py3-sdk
  ```
  - Install the requirements (for visualization)
  ```
  pip install -r requirements
  ```
  - Start jupyter lab
  ```
  jupyter lab --ip 0.0.0.0 --allow-root
  ```
  - Triton Client SDK
    -  Inference
    -  Performance Analyzer
