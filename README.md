# TensorRT Benchmark
"NVIDIA® TensorRT™, an SDK for high-performance deep learning inference, includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for inference applications." https://developer.nvidia.com/tensorrt </br>

These experimets are based on <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch">nvcr.io/nvidia/pytorch:23.04-py3</a> docker image. The environment and notebooks are also packaged as a <a href="https://hub.docker.com/r/warrents/trt_benchmark">docker image</a>. </br>

Please follow the instructions below to start the testing environment
1. Pull the docker image
```
docker pull warrents/trt_benchmark:ngc_torch-23.04-py3
```
2. Start the environment and jupyter lab
```
docker run -it --rm --shm-size=2g -p 8888:8888 --gpus='"device=0"' warrents/trt_benchmark:ngc_torch-23.04-py3
jupyter lab --ip 0.0.0.0 --allow-root
```

## Results

### For Classification (DenseNet201, Batch Size=256)

| GPU  | Torch Script </br> Throughputs (imgs/sec) | TensorRT-FP16 </br> Throughputs (imgs/sec) | Energy Efficiency </br> for TRT-FP16 </br> (throughputs/watt) |
| :-------------: |:-------------:|:-------------:|:-------------:|
| V100 - 16GB     | 463.76        | 3622.02       | 22.64         |
| L4              | 333.11        | ***5140.12*** | ***71.39***   |
| A6000           | ***806.12***  | 4210.68       | 14.04         |
</br></br>
### For Detection (Yolov5s, Batch Size=256)

| GPU  | Native PyTorch </br> Throughputs (imgs/sec) | TensorRT-FP16 </br> Throughputs (imgs/sec) | Energy Efficiency </br> for TRT-FP16 </br> (throughputs/watt) |
| :-------------: |:-------------:|:-------------:|:-------------:|
| V100 - 16GB     | 532.92        | 4085.98       | 25.54         |
| L4              | ***925.46***  | ***6698.22*** | ***93.03***   |
| A6000           | 904.82        | 4603.43       | 15.34         |
</br></br>
### For Segmentation (SegResNet, Batch Size=4)

| GPU  | Native PyTorch </br> Throughputs (imgs/sec) | TensorRT-FP16 </br> Throughputs (imgs/sec) | Energy Efficiency </br> for TRT-FP16 </br> (throughputs/watt) |
| :-------------: |:-------------:|:-------------:|:-------------:|
| V100 - 16GB     | 2.16          | 22.59         | 0.141         |
| L4              | 2.41          | 19.75         | ***0.274***   |
| A6000           | ***4.37***    | ***29.62***   | 0.099         |
| A6000 (bs=128)  | OOM           | *138.50       | *0.462        |
</br></br>

### Details
#### V100 - with 16GB RAM, 160 watt
| Task           | Batch Size | Native PyTorch </br> (imgs/sec) | Torch Script </br> (imgs/sec) |TensorRT-FP32 </br> (imgs/sec) | TensorRT-FP16 </br> (imgs/sec)| Improvement </br> (TRT v.s. PT) | Energy Efficiency </br>for TRT-FP16 </br> (throughputs/watt) |
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Classification </br> (DenseNet201) | 32  | 400.29 | 479.55 | 607.51  | 1300.60 | 3.25  | 8.13  |
| Classification </br> (DenseNet201) | 256 | 397.40 | 463.76 | 2654.23 | 3622.02 | 9.11  | 22.64 |
| Detection </br> (Yolov5s)          | 32  | 894.63 | -      | 1583.10 | 1947.37 | 2.18  | 12.17 |
| Detection </br> (Yolov5s)          | 256 | 532.92 | -      | 3764.73 | 4085.98 | 7.67  | 25.54 |
| Segmentation </br> (SegResNet)     | 1   | 1.96   | 1.84   | 2.64    | 6.26    | 3.19  | 0.039 |
| Segmentation </br> (SegResNet)     | 4   | 2.19   | 2.16   | 10.08   | 22.59   | 10.32 | 0.141 |

#### A2 - with 16GB RAM, 60 watt (40-60 watt configurable)
| Task           | Batch Size | Native PyTorch </br> (imgs/sec) | Torch Script </br> (imgs/sec) | TensorRT-FP32 </br> (imgs/sec) | TensorRT-FP16 </br> (imgs/sec)| Improvement </br> (TRT v.s. PT) | Energy Efficiency </br>for TRT-FP16 </br> (throughputs/watt) |
|  ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Classification </br> (DenseNet201) | 32  |   |  |    |  |   |  |
| Classification </br> (DenseNet201) | 256 |   |  |   |  |  |  |
| Detection </br> (Yolov5s)          | 32  |  |       |   |  |   |  |
| Detection </br> (Yolov5s)          | 256 |   |       |   |  |   |  |
| Segmentation </br> (SegResNet)     | 1   |     |    |      |     |   |  |
| Segmentation </br> (SegResNet)     | 4   |     |    |     |    |   |  |

#### L4 - with 24GB RAM, 72 watt
| Task           | Batch Size | Native PyTorch </br> (imgs/sec) | Torch Script </br> (imgs/sec) | TensorRT-FP32 </br> (imgs/sec) | TensorRT-FP16 </br> (imgs/sec)| Improvement </br> (TRT v.s. PT) | Energy Efficiency </br>for TRT-FP16 </br> (throughputs/watt) |
|  ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Classification </br> (DenseNet201) | 32  | 313.77  | 410.30 | 590.43   | 1351.51 | 4.31  | 18.77 |
| Classification </br> (DenseNet201) | 256 | 240.96  | 333.11 | 3198.02  | 5140.12 | 21.33 | 71.39 |
| Detection </br> (Yolov5s)          | 32  | 1221.17 | -      | 2102.50  | 3076.79 | 2.52  | 42.73 |
| Detection </br> (Yolov5s)          | 256 | 925.46  | -      | 5960.74  | 6698.22 | 7.24  | 93.03 |
| Segmentation </br> (SegResNet)     | 1   | 2.08    | 2.07   | 3.07     | 5.19    | 2.50  | 0.072 |
| Segmentation </br> (SegResNet)     | 4   | 2.41    | 2.41   | 11.88    | 19.75   | 8.20  | 0.274 |

#### A6000 - with 48GB RAM, 300 watt
| Task           | Batch Size | Native PyTorch </br> (imgs/sec) | Torch Script </br> (imgs/sec) | TensorRT-FP32 </br> (imgs/sec) | TensorRT-FP16 </br> (imgs/sec)| Improvement </br> (TRT v.s. PT) | Energy Efficiency </br>for TRT-FP16 </br> (throughputs/watt) |
|  ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Classification </br> (DenseNet201) | 32  | 586.77  | 774.25 | 1014.38   | 1687.10 | 2.88  | 5.62  |
| Classification </br> (DenseNet201) | 256 | 617.60  | 806.12 | 3516.77   | 4210.68 | 6.82  | 14.04 |
| Detection </br> (Yolov5s)          | 32  | 874.32  | -      | 2093.72   | 2590.53 | 2.96  | 8.64  |
| Detection </br> (Yolov5s)          | 256 | 904.82  | -      | 4380.59   | 4603.43 | 5.09  | 15.34 |
| Segmentation </br> (SegResNet)     | 1   | 3.34    | 3.33   | 5.65      | 8.64    | 2.59  | 0.029 |
| Segmentation </br> (SegResNet)     | 4   | 4.39    | 4.37   | 20.38     | 29.62   | 6.75  | 0.099 |
| Segmentation </br> (SegResNet)     | 8   | 4.58    | 4.56   | 36.06     | 49.98   | 10.91 | 0.167 |
| Segmentation </br> (SegResNet)     | 16  | OOM     | 4.66   | 58.60     | 75.74   | -     | 0.252 |
| Segmentation </br> (SegResNet)     | 32  | OOM     | OOM    | 85.01     | 102.35  | -     | 0.341 |
| Segmentation </br> (SegResNet)     | 128 | OOM     | OOM    | 129.05    | 138.50  | -     | 0.462 |
