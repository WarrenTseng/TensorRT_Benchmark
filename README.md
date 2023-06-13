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
  - Classification
  - Detection
  - Segmentation

## TensorRT with Triton Inference Server
To compare TorchScript with TensorRT in Triton, we need the environments as below:
- PyTorch: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch  
  - nvcr.io/nvidia/pytorch:23.04-py3
- Triton Server and Client: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
  - Server: nvcr.io/nvidia/tritonserver:23.04-py3
  - Client: nvcr.io/nvidia/tritonserver:23.04-py3-sdk 

Please follow the instructions below to start the testing environment:
1. Pull the docker images:
```
docker pull nvcr.io/nvidia/pytorch:23.04-py3
docker pull nvcr.io/nvidia/tritonserver:23.04-py3
docker pull nvcr.io/nvidia/tritonserver:23.04-py3-sdk
```
2. Preparing the models and Triton configs:
  2.1. Run the environment:
  ```
  docker run -it --rm --shm-size=2g -p 8888:8888 --gpus='"device=0"' -v /PATH/AS/TRITON/REPO:/repo nvcr.io/nvidia/pytorch:23.04-py3
  ```
  2.2 



