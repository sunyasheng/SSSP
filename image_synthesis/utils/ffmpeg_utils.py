import os
import cv2
from tqdm import tqdm
import glob

def img2video(dst_path, prefix, video_path):
    cmd = ['ffmpeg', '-i', '\'' + video_path + '/' + prefix + '/%06d.jpg'
           + '\'', '-q:v 0', '\'' + dst_path + '/' + prefix + '.mp4' + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)


def frame_dir2video(root, prefix, frame_dir, video_format='mp4v', start_number=2):
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, prefix, '*.jpg')))
    if start_number > 0: frame_paths = frame_paths[start_number:]
    if start_number < 0:
        start_number = -start_number
        frame_paths = frame_paths[:start_number] + frame_paths
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]
    width, height = frames[0].shape[1], frames[0].shape[0]
    tgt_mp4_path = os.path.join(root, prefix+'.mp4')

    fourcc = cv2.VideoWriter_fourcc(*video_format) 
    video = cv2.VideoWriter(tgt_mp4_path, fourcc, 25, (width, height))
    
    for i, frame in enumerate(frames):
        video.write(frame)

    # cv2.destroyAllWindows()
    video.release()

def video_concat(processed_file_savepath, name, video_names, audio_path):
    cmd = ['ffmpeg']
    num_inputs = len(video_names)
    for video_name in video_names:
        cmd += ['-i', '\'' + str(os.path.join(processed_file_savepath, video_name + '.mp4'))+'\'',]

    cmd += ['-filter_complex hstack=inputs=' + str(num_inputs),
            '\'' + str(os.path.join(processed_file_savepath, name+'.mp4')) + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)

    video_add_audio(name, audio_path, processed_file_savepath)


def video_add_audio(name, audio_path, processed_file_savepath):
    os.system('cp {} {}'.format(audio_path, processed_file_savepath))
    cmd = ['ffmpeg', '-i', '\'' + os.path.join(processed_file_savepath, name + '.mp4') + '\'',
                     '-i', audio_path,
                     '-q:v 0',
                     '-strict -2',
                     '\'' + os.path.join(processed_file_savepath, 'av' + name + '.mp4') + '\'',
                     '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)

    cmd = 'rm -rf {}'.format(os.path.join(processed_file_savepath, name + '.mp4'))
    os.system(cmd)

    driven_video_path = audio_path.replace('.wav', '.mp4')
    if os.path.exists(driven_video_path):
        cmd = 'cp {} {}'.format(driven_video_path, processed_file_savepath)
        os.system(cmd)


def rm_imgs(root_dir, prefix=''):
    cmd = 'rm -rf {}/{}/*.jpg'.format(root_dir, prefix)
    os.system(cmd)

    cmd = 'rm -rf {}/{}/*.png'.format(root_dir, prefix)
    os.system(cmd)


def get_frames(videoPath):
    cap = cv2.VideoCapture(videoPath)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    forcc = cap.get(cv2.CAP_PROP_FOURCC)
    # print("Video info\n-> W x H: {} x {}\n-> FPS: {}\n-> CODE: {}".format(w, h, fps, forcc))

    if not cap.isOpened():
        print("Fail to open '{}'".format(videoPath))
        return -1
    
    frames = []
    ret, img = cap.read()
    while(ret == True):
        frames.append(img)
        ret, img = cap.read()
    cap.release()
    return frames

def get_frame_num(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(frame_num)


def video2imgs(video_path, video_dir):
    os.system('rm -rf {}'.format(video_dir))
    os.makedirs(video_dir, exist_ok=True)
    # cmd = 'ffmpeg -i {} -q:v 0 -start_number 0 {}/%06d.jpg -loglevel error -y'.format(video_path, video_dir)
    # # print(cmd)
    # os.system(cmd)
    # frame_num = get_frame_num(video_path)
    frames = get_frames(video_path)
    for i, frame in enumerate(frames):
        name = '{:06d}.jpg'.format(i)
        path = os.path.join(video_dir, name)
        cv2.imwrite(path, frame)
    # print('=============================')
    # print('frame_num: ', frame_num)
