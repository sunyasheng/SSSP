import cv2
import os
from tqdm import tqdm


def write2video(mp4_align_path, mp4_affine_frames, width=512, height=512, video_format='mp4v'):
    os.makedirs(os.path.dirname(mp4_align_path), exist_ok=True)

    width, height = mp4_affine_frames[0].shape[1], mp4_affine_frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*video_format) 
    video = cv2.VideoWriter(mp4_align_path, fourcc, 25, (width, height))
    
    for i, mp4_affine_frame in tqdm(enumerate(mp4_affine_frames)):
        video.write(mp4_affine_frame)

    # cv2.destroyAllWindows()
    video.release()