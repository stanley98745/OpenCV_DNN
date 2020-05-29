########################################################################
#
#  Copyright (C) 2020. All rights reserved
#
#  Author       : Stanley J. F. Zhang
#  Created Date : 2020-05-29
#  Description  : Classify the dog image using yolov3 and tensorflow.
#                 The performance will be compared with opencv dnn
#                 module.
#  Reference    : https://github.com/YunYang1994/tensorflow-yolov3
#
########################################################################


from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import time
import cv2

import core.utils as utils

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", required=True,
                    help="path to yolo-coco folder")
    ap.add_argument("-i", "--image", required=True,
                    help="path to image file")
    ap.add_argument("-g", "--gpu_utility", type=float, default=0.5,
                    help="reduce gpu memory utility to prevent from KILLED")
    ap.add_argument("-t", "--times", type=int, required=True, default=100,
                    help="testing loop for image")
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
    image_data = utils.image_preporcess(np.copy(original_image),
                                        [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    '''Tensorflow setting'''
    graph = tf.Graph()
    gpu_utility = tf.GPUOptions(per_process_gpu_memory_fraction=args["gpu_utility"])
    config = tf.ConfigProto(gpu_options=gpu_utility)
    return_tensors = utils.read_pb_return_tensors(graph,
                                                  pb_file,
                                                  return_elements)

    '''LOOP'''
    with tf.Session(config=config, graph=graph) as sess:
        start = time.time()
        for i in range(args["times"]):
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={return_tensors[0]: image_data})
        end = time.time()

    if args["display"] is True:
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.5, method='nms')
        image = utils.draw_bbox(original_image, bboxes)
        image = Image.fromarray(image)
        image.show()

    ms_per_image = (end - start) * 1000/100
    print(f'Time per inference: {ms_per_image} ms')
    print(f'FPS: {1000/ms_per_image}')
