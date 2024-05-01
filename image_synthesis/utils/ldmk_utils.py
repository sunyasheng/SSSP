import cv2
import numpy as np


def draw_lines(seq_points, bg, color=(255,255,255), thickness=2):
    for i in range(len(seq_points)-1):
        start_point = seq_points[i]
        end_point = seq_points[i+1]
        # import pdb; pdb.set_trace();
        bg = cv2.line(bg, start_point, end_point, color, thickness)
    return bg


def draw_ldmk(ldmk_np, image_shape, thickness=2, color=(255,255,255)):
    bg = np.zeros(shape=image_shape)
    # contour
    bg = draw_lines(ldmk_np[0:17], bg, color=color, thickness=thickness)
    # nose
    bg = draw_lines(ldmk_np[31:36], bg, color=color, thickness=thickness)
    bg = draw_lines(ldmk_np[27:31], bg, color=color, thickness=thickness)
    # eyebrows
    bg = draw_lines(ldmk_np[17:22], bg, color=color, thickness=thickness)
    bg = draw_lines(ldmk_np[22:27], bg, color=color, thickness=thickness)
    # mouth
    bg = cv2.polylines(bg,[ldmk_np[48:60].reshape(-1,1,2)],True,color, thickness=thickness)
    bg = cv2.polylines(bg,[ldmk_np[60:69].reshape(-1,1,2)],True,color, thickness=thickness)
    # eye
    bg = cv2.polylines(bg,[ldmk_np[36:42].reshape(-1,1,2)],True,color, thickness=thickness)
    bg = cv2.polylines(bg,[ldmk_np[42:48].reshape(-1,1,2)],True,color, thickness=thickness)
    
    return bg