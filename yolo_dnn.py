########################################################################
#
#  Copyright (C) 2020. All rights reserved
#
#  Author       : Stanley J. F. Zhang
#  Created Date : 2020-05-29
#  Description  : Classify the image using yolov3 and opencv DNN.
#                 The performance will be compared with tensorflow.
#
#  Reference    : https://github.com/Cuda-Chen/opencv-dnn-cuda-test
#                 https://github.com/YunYang1994/tensorflow-yolov3
#
########################################################################


import numpy as np
import argparse
import time
import cv2


def imgpreprocess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", default="./yolo-coco",
                    help="path to yolo weight, cfg folder")
    ap.add_argument("-i", "--image", default="./doc/image/dog.jpg",
                    help="path to image file")
    ap.add_argument("-t", "--times", type=int,
                    help="loop for testing")
    args = vars(ap.parse_args())

    yoloWeights = args["yolo"] + "/yolov3.weights"
    yoloConfig = args["yolo"] + "/yolov3.cfg"

    num_classes = 80
    input_size = 416

    net = cv2.dnn.readNetFromDarknet(yoloConfig, yoloWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    '''Reading Image'''
    original_image = cv2.imread(args["image"])
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = imgpreprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]
    blob = cv2.dnn.blobFromImage(original_image, 0.00392, (416, 416), [0, 0, 0], True, False)

    '''Benchmarking'''
    start = time.time()
    for i in range(args["times"]):
        net.setInput(blob)
        pred_sbbox, pred_mbbox, pred_lbbox = net.forward(ln)
    end = time.time()

    ms_per_image = (end - start) * 1000 / args["times"]
    print(f'Time per inference: {ms_per_image} ms')
    print(f'FPS: {1000 / ms_per_image}')
