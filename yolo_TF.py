########################################################################
#
#  Copyright (C) 2020. All rights reserved
#
#  Author       : Stanley J. F. Zhang
#  Created Date : 2020-05-29
#  Description  : Classify the image using yolov3 and tensorflow.
#                 The performance will be compared with opencv dnn
#                 module.
#  Reference    : https://github.com/Cuda-Chen/opencv-dnn-cuda-test
#                 https://github.com/YunYang1994/tensorflow-yolov3
#
########################################################################


import numpy as np
import tensorflow as tf
import argparse
import time
import cv2


def imgpreprocess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)
    return return_elements


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", default="./yolo-coco",
                    help="path to yolo-coco.pb folder")
    ap.add_argument("-i", "--image", default="./doc/image/dog.jpg",
                    help="path to image file")
    ap.add_argument("-g", "--gpu_utility", type=float, default=0.5,
                    help="reduce gpu memory utility to prevent from KILLED")
    ap.add_argument("-t", "--times", type=int, default=100,
                    help="loop for testing")
    args = vars(ap.parse_args())

    '''Parameter setting'''
    pb_file = args["yolo"] + "/yolov3_coco.pb"
    # image_path = "./docs/images/road.jpeg"
    num_classes = 80
    input_size = 416
    return_elements = ["input/input_data:0",
                       "pred_sbbox/concat_2:0",
                       "pred_mbbox/concat_2:0",
                       "pred_lbbox/concat_2:0"]

    '''Reading Image'''
    original_image = cv2.imread(args["image"])
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = imgpreprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    '''Tensorflow setting'''
    graph = tf.Graph()
    gpu_utility = tf.GPUOptions(per_process_gpu_memory_fraction=args["gpu_utility"])
    config = tf.ConfigProto(gpu_options=gpu_utility)
    return_tensors = read_pb_return_tensors(graph, pb_file, return_elements)

    '''LOOP'''
    with tf.Session(config=config, graph=graph) as sess:
        start = time.time()
        for i in range(args["times"]):
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={return_tensors[0]: image_data})
        end = time.time()

    ms_per_image = (end - start) * 1000/args["times"]
    print(f'Time per inference: {ms_per_image} ms')
    print(f'FPS: {1000/ms_per_image}')
