# OpenCV_DNN on Jetson

For someone who is interested in OpenCV DNN module and want to applied it on Jetson device. 

> device: Jetson AGX Xavier
>
> operating system (os): Ubuntu

<br/>

## How to use

Because I want to compare the performance in GPU. so you should install following packages.

<br/>

**required packages:**

- OpenCV DNN CUDA
- Tensorflow GPU

If you don't know how to install **OpenCV DNN CUDA**, check these

[Jstat](https://github.com/rbonghi/jetson_stats)

[nano_build_opencv](https://github.com/mdegans/nano_build_opencv)


### parameter setting

**general**

``` -y, --yolo: path to yolo weight, cfg, pb folder; default: ./yolo-coco```

``` -i, --image: path to image file; default: ./doc/image/dog.jpg```

``` -t, --times: loop for testing; default: 100```

<br/>

**only for yolo_TF.py**

``` -g, --gpu_utility: reduce gpu memory utility to prevent from KILLED; default: 0.5```

<br/>

## Benchmark: OpenCV DNN CUDA vs. Tensorflow GPU

### Tensorflow GPU Perfoemance

```python3 yolo_TF.py```

or use your own setting

```python3 yolo_TF.py --yolo ./yolo-coco --image ./doc/images/dog.jpg --gpu_utility 0.25 --times 100```

### OpenCV DNN CUDA Perfoemance

```python3 yolo_dnn.py```

or use your own setting

```python3 yolo_dnn.py --yolo ./yolo-coco --image ./doc/images/dog.jpg --times 100``` 