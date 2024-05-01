import os
from tqdm import tqdm
import numpy as np
# from image_utils import psnr
# import face_alignment
import cv2
import torch
from skimage.metrics import structural_similarity as ssim
import torchvision


def draw_outlier_results(root, preds, gts, scores, thr):
    os.makedirs(root, exist_ok=True)
    for i, score in enumerate(scores):
        if score < thr:
            cat = torch.cat([gts[i], preds[i]], dim=-2)
            outliner_path = os.path.join(root, '{:05d}.jpg'.format(i))
            torchvision.utils.save_image(cat, outliner_path)



def cal_sketch_ssim(pred_ts_frames, gt_ts_frames, image2simplified_sketcher, arcface_model):
    pred_ts_frames = [ts.permute(2,0,1).unsqueeze(0)/255. for ts in pred_ts_frames]
    gt_ts_frames = [ts.permute(2,0,1).unsqueeze(0)/255. for ts in gt_ts_frames]
    
    pred_ts_sketches = [image2simplified_sketcher(ts.cuda()) for ts in pred_ts_frames]
    gt_ts_sketches = [image2simplified_sketcher(ts.cuda()) for ts in gt_ts_frames]

    # import torchvision
    # torchvision.utils.save_image(torch.cat([pred_ts_frames[0], gt_ts_frames[0]], dim=-2), 'pred_gt_img.jpg')
    # torchvision.utils.save_image(torch.cat([pred_ts_sketches[0], gt_ts_sketches[0]], dim=-2), 'pred_gt_sketch.jpg')
    # import pdb; pdb.set_trace();

    pred_np_sketches = [ts.squeeze().cpu().numpy() for ts in pred_ts_sketches]
    gt_np_sketches = [ts.squeeze().cpu().numpy() for ts in gt_ts_sketches]

    ssims = [ssim(pred_np_sketch, gt_np_sketch, multichannel=True) \
            for pred_np_sketch, gt_np_sketch in zip(pred_np_sketches, gt_np_sketches) ]

    draw_outlier_results('output_w_lower_ssim', pred_ts_sketches, gt_ts_sketches, ssims, thr=0.6)
    mean_ssim = np.mean(ssims)
    return mean_ssim


def cal_id_similarity(pred_ts_frames, gt_ts_frames, image2simplified_sketcher, arcface_model):
    pred_ts_frames = [ts.permute(2,0,1).unsqueeze(0)/255. for ts in pred_ts_frames]
    gt_ts_frames = [ts.permute(2,0,1).unsqueeze(0)/255. for ts in gt_ts_frames]
    
    # import pdb; pdb.set_trace();
    similarities = [torch.sum(arcface_model(gt.cuda())*arcface_model(pred.cuda())) for gt, pred in zip(gt_ts_frames,pred_ts_frames)]
    similarities = [s.cpu().numpy() for s in similarities]

    draw_outlier_results('output_w_large_dissimilarity/', pred_ts_frames, gt_ts_frames, similarities, thr=0.3)

    mean_similarity = np.mean(similarities)
    return mean_similarity