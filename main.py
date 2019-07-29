#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
import json
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
from collections import OrderedDict
t1_res_cai = OrderedDict()
ress = []

def present_result(num_id, num_person, num_fire_extinguisher, num_fireplug, num_car, num_bicycle, num_motocycle):
	t1_res = OrderedDict()
	t1_res["id"] = num_id
	t1_res["objects"] = [num_person,
						num_fire_extinguisher,
						num_fireplug,
						num_car,
						num_bicycle,
						num_motocycle]
	return t1_res
backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/det_t1_video_00046_test.avi")
ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
#list = [[] for _ in range(100)]

def main():
    start = time.time()
    first = start
    #Definition of the parameters
    max_cosine_distance = 0.5#0.9 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3 #非极大抑制的阈值

    counter = []

    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    find_objects = ['person', 'fire_extinguisher', 'fireplug', 'car', 'bicycle', 'motorcycle']
    yolo = YOLO()

    for cnt in range(1, 2):
        video_path = "./t1_video/t1_video_%05d" % cnt
        images = os.listdir(video_path)
        images.sort()
        print(images[0])
        trackers = []
        counters = []
        for idx in range(0, 6):
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            trackers.append(Tracker(metric))
            count = []
            counters.append(count)
        tracker_time = 0
        yolo_time = 0
        for image_path in images:
            # image_path = video_path + "/t1_video_%05d_%05d.jpg" % (1, fc)
            t1 = time.time()
            # print(video_path + "/" + image_path)
            frame = cv2.imread(video_path + "/" + image_path)
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb

            yolo_start = time.time()
            yolo_dict = yolo.detect_image(image)
            yolo_end = time.time()
            yolo_time += (yolo_end - yolo_start)

            for idx in range(0, 6):
                # print(idx)
                tracker = trackers[idx]
                counter = counters[idx]

                boxs = yolo_dict.get(find_objects[idx])
                if boxs == None:
                    continue

                features = encoder(frame, boxs)
                # score to 1.0 here).
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                t_start = time.time()
                tracker.predict()
                tracker.update(detections)
                t_end = time.time()
                tracker_time += (t_end - t_start)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    #boxes.append([track[0], track[1], track[2], track[3]])

                    counter.append(int(track.track_id))
        #######################################
        num_person = len(set(counters[0]))
        num_fire_extinguisher = len(set(counters[1]))
        num_fireplug = len(set(counters[2]))
        num_car = len(set(counters[3]))
        num_bicycle = len(set(counters[4]))
        num_motocycle = len(set(counters[5]))
        ress.append(present_result(cnt, num_person,
                            num_fire_extinguisher,
							num_fireplug,
							num_car,
							num_bicycle,
							num_motocycle))

        t1_res_cai["track1_results"] = ress
        with open('t1_res_cai.json', 'w') as make_file:
            json.dump(t1_res_cai, make_file, ensure_ascii=False, indent=4)
		#######################################


        for idx in range(0, 6):
            print(len(set(counters[idx])), end=" ")
        end = time.time()
        print(str(':: total:%.2f yolo:%.2f tracker:%.2f' % ((end - start),yolo_time, tracker_time)))
        start = end
    last = time.time()
    print(str(':: %.2f' % (last - first)))

if __name__ == '__main__':
    main()
