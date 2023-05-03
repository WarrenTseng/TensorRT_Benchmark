# TensorRT_Benchmark

These experimets are based on nvcr.io/nvidia/pytorch:23.04-py3 docker image. The environment and notebooks are also packaged as a <a href="https://hub.docker.com/repository/docker/warrents/trt_benchmark">docker image</a>. </br>

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
