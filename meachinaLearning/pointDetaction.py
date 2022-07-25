#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard imports
import os

import cv2
import numpy as np

files = os.listdir('./imgs')

for file in files:

    # Read image
    im = cv2.imread('./imgs/{}'.format(file))
    im_o = cv2.resize(im, (800, 600))
    #
    # im_gauss = cv2.cvtColor(im_o, cv2.COLOR_RGB2GRAY)
    # im_gauss = cv2.GaussianBlur(im_gauss, (3, 3), 0)
    # ret, im = cv2.threshold(im_gauss, 100, 255, 0)
    # cv2.imwrite("o.png", im)
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 16

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.57

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im_o)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(im_o, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imwrite("./res/{}".format(file), im_with_keypoints)
    # cv2.waitKey(0)
