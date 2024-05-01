import cv2
import numpy as np


def contour_to_dist(contour):
    invert_contour = cv2.bitwise_not(np.array(contour))
    dist = cv2.distanceTransform(invert_contour, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist = dist * 255
    return dist
