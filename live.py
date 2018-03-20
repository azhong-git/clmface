#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import multiprocessing
import numpy as np
import six
import cv2
import time

from extensions import imgliveuploader
import matplotlib.pyplot as plt

from clmface.tracker import clmFaceTracker

# logging
from logging import getLogger, DEBUG, INFO
logger = getLogger(__name__)
# logging for imgliveuploader
imgliveuploader.logger.setLevel(INFO)

if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()

    logger.info('Start visualization')

    # Start ImgLiveUploader
    request_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()
    imgliveuploader.start(request_queue, response_queue, stop_page=False,
                          port=9000)

    # initialize face tracker
    tracker = clmFaceTracker('models/model_pca_20_svm.json', debug=False)
    count = 0

    # Main loop
    logger.info('Start main loop')
    while True:
        img = request_queue.get(timeout=60.0)
        start_time = time.time()
        # BGR to gray
        # floating point precision but slower > 1ms
        # gray=img[:,:,0]*0.11+img[:,:,1]*0.59+img[:,:,2]*0.3
        # print gray.shape
        # faster but integer precision 0.2ms
        gray = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        logger.info('Next image')
        count += 1

        converged, score = tracker.track(gray)

        if tracker.currentPositions is not None:
            for x, y in tracker.currentPositions:
                cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255))

            if tracker.config.localTracking:
                for rect in tracker.localTrackerRects:
                    p1 = (int(rect[0]), int(rect[1]))
                    p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
                    cv2.rectangle(img, p1, p2, (0, 0, 255))

                if tracker.localTrackerUpdate:
                    cv2.putText(img, 'tracker: %.2f, %.2f, %.5f, %.5f' % (tracker.localTrackerUpdate[0],
                                                                          tracker.localTrackerUpdate[1],
                                                                          tracker.localTrackerUpdate[2],
                                                                          tracker.localTrackerUpdate[3]),
                                (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))


            if score < 0.4:
                cv2.putText(img, '%.2f out of %.2f' % (score, tracker.max_face_score), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            else:
                cv2.putText(img, '%.2f out of %.2f' % (score, tracker.max_face_score), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))



        latency_in_ms = (time.time() - start_time) * 1e3
        print('one frame takes {} ms'.format(latency_in_ms))

        response_queue.put({'img': img}, timeout=1.0)
