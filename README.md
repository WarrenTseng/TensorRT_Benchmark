# TensorRT Benchmark
"NVIDIA® TensorRT™, an SDK for high-performance deep learning inference, includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for inference applications." https://developer.nvidia.com/tensorrt </br>

These experimets are based on <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch">nvcr.io/nvidia/pytorch:23.04-py3</a> docker image. The environment and notebooks are also packaged as a <a href="https://hub.docker.com/repository/docker/warrents/trt_benchmark">docker image</a>. </br>

Please follow the instructions below to start the testing environment
1. Pull the docker image
```
docker pull warrents/trt_benchmark:ngc_torch-23.04-py3
```
2. Start the environment and jupyter lab
```
docker run -it --rm -p 8888:8888 --gpus='"device=0"' warrents/trt_benchmark:ngc_torch-23.04-py3
jupyter lab --ip 0.0.0.0 --allow-root
```
