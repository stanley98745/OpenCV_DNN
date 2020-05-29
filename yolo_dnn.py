########################################################################
#
#  Copyright (C) 2020. All rights reserved
#
#  Author       : Stanley J. F. Zhang
#  Created Date : 2020-05-29
#  Description  : Classify the dog image using yolov3 and opencv DNN.
#                 The performance will be compared with tensorflow.
#
#  Reference    : https://github.com/Cuda-Chen/opencv-dnn-cuda-test
#
########################################################################


from PIL import Image
import numpy as np
import argparse
import time
import cv2

import core.utils as utils

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", required=True,
                    help="path to yolo weight, cfg folder")
    ap.add_argument("-i", "--image", required=True,
                    help="path to image file")
    ap.add_argument("-t", "--times", type=int, default=100,
                    help="testing loop for image")
    args = vars(ap.parse_args())

    yoloWeights = args["yolo"] + "yolov3.weights"
    yoloConfig = args["yolo"] + "yolov3.cfg"

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
    image_data = utils.image_preporcess(np.copy(original_image),
                                        [input_size, input_size])
    image_data = image_data[np.newaxis, ...]
    blob = cv2.dnn.blobFromImage(original_image, 0.00392, (416, 416), [0, 0, 0], True, False)

    # benchmark
    start = time.time()
    for i in range(args["times"]):
        net.setInput(blob)
        pred_sbbox, pred_mbbox, pred_lbbox = net.forward(ln)
    end = time.time()

    ms_per_image = (end - start) * 1000/args["time"]
    print(f'Time per inference: {ms_per_image} ms')
    print(f'FPS: {1000/ms_per_image}')

