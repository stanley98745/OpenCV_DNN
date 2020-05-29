# OpenCV_DNN on Jetson

For someone who is interested in OpenCV DNN module and want to applied it on Jetson device. 

> device: Jetson AGX Xavier
>
> operating system (os): Ubuntu

<br/>

## Benchmark: OpenCV DNN CUDA vs. Tensorflow GPU

Because I want to compare the performance in GPU. so you should install following packages.

<br/>

**required packages:**

- OpenCV DNN CUDA
- Tensorflow GPU

If you don't know how to install **OpenCV DNN CUDA**, check these

[Jstat](https://github.com/rbonghi/jetson_stats)

[nano_build_opencv](https://github.com/mdegans/nano_build_opencv)


<br/>

### Tensorflow GPU Perfoemance

```python3 yolo_TF.py --yolo ./yolo-coco --image ./doc/images/dog.jpg --times 100```

### OpenCV DNN CUDA Perfoemance

```python3 yolo_dnn.py --yolo ./yolo-coco --image ./doc/images/dog.jpg --times 100``` 
