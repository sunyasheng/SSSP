import argparse
import cv2
import os
from tqdm import tqdm
import numpy as np
import math
# import face_alignment
# from metric_collections import cal_cpbd_video, cal_lmd_video, cal_psrn_video, cal_ssim_video, cal_cpbd_tgt_video
from tqdm import tqdm
from metric_collections import cal_sketch_ssim, cal_id_similarity
import glob
import torch
from image_synthesis.modeling.modules.img_operators import Image2SimplifiedSketch
from image_synthesis.modeling.modules.arcface_operators import define_net_recog


def compute_metric_by_func(res_root, gt_root, names, fncs, image2simplified_sketcher, arcface_model):
    # pred_frames, tgt_frames = [], []
    pred_frame_paths = glob.glob(os.path.join(res_root, '*.jpg'))
    pred_frame_paths += glob.glob(os.path.join(res_root, '*.png'))
    gt_frame_paths = [p.replace(res_root, gt_root) for p in pred_frame_paths]
    
    pred_np_frames = [cv2.cvtColor(cv2.imread(pred_frame_path), cv2.COLOR_BGR2RGB) for pred_frame_path in pred_frame_paths]
    gt_np_frames = [cv2.cvtColor(cv2.imread(gt_frame_path), cv2.COLOR_BGR2RGB) for gt_frame_path in gt_frame_paths]
    pred_ts_frames = [torch.from_numpy(pred_np_frame) for pred_np_frame in pred_np_frames]
    gt_ts_frames = [torch.from_numpy(gt_np_frame) for gt_np_frame in gt_np_frames]

    res = dict()
    for name,fnc in zip(names, fncs):
        res_i = fnc(pred_ts_frames, gt_ts_frames, image2simplified_sketcher, arcface_model)
        res[name] = res_i
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_root', type=str, default='')
    parser.add_argument('--gt_root', type=str, default='') # gt frames may related to pred frames one by one
    parser.add_argument('--net_recog', type=str, default='r50')
    parser.add_argument('--pretrained_path', type=str, default='OUTPUT/pretrained/backbone.pth')
    args = parser.parse_args()

    metrics = {
                'sketch_ssim': cal_sketch_ssim,
               'id_similarity': cal_id_similarity,
               }
    
    res_dict = {
                'sketch_ssim': [],
                'id_similarity': [],
                }

    image2simplified_sketcher = Image2SimplifiedSketch()
    image2simplified_sketcher = image2simplified_sketcher.cuda()
    
    arcface_model = define_net_recog(args.net_recog, args.pretrained_path)
    arcface_model = arcface_model.cuda()

    try:
        res_i = compute_metric_by_func(args.res_root, args.gt_root, metrics.keys(), metrics.values(), image2simplified_sketcher, arcface_model)
        for key, value in res_i.items():
            res_dict[key].append(res_i[key])
    except Exception as ex:
        import traceback
        traceback.print_exc()

    print_line = ''
    for key, value in res_dict.items():
        print_line += '{}: {:.6f} || '.format(key, np.mean(res_dict[key]))

    print(print_line)


if __name__ == '__main__':
    main()
