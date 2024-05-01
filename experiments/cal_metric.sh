set -ex

jobname=$1
res_root=$2
gt_root='/root/dataset/FFHQ/test_qual/CROP'



run_consistency_eval() {
    res_root=$1
    gt_root=$2
    python metrics/cal_metric.py --res_root ${res_root} \
                        --gt_root ${gt_root}
}

run_fid_eval() {
    cd metrics
    res_root=$1
    gt_root=$2
    batch_size=64
    num_workers=8
    dims=2048

    python3.8 pytorch_fid/fid_score.py \
                --batch-size ${batch_size} \
                --num-workers ${num_workers} \
                --dims ${dims} \
                --path1 ${res_root} \
                --path2 ${gt_root}
}

run_is_eval() {
    res_root=$1
    python3.8 metrics/inception_score.py \
        --path ${res_root}
}


if [[ ${jobname} == 'consistency' ]]; then
    res_root='/root/Proj/Sketch2Nerf/OUTPUT/test/'${jobname}'_pred_render'
    gt_root='/root/dataset/FFHQ/train/CROP'
    run_consistency_eval ${res_root} ${gt_root}
fi



if [[ ${jobname} == 'fid' ]]; then
    # res_root='/root/Proj/Sketch2Nerf/OUTPUT/test/sketch128'
    # gt_root='/root/dataset/FFHQ/train/CROP'
    run_fid_eval ${res_root} ${gt_root}
fi



if [[ ${jobname} == 'is' ]]; then
    # res_root='/root/Proj/Sketch2Nerf/OUTPUT/test/sketch128'
    run_is_eval ${res_root}
fi



### bash experiments/cal_metric.sh fid /root/Proj/DeepFaceDrawing-Jittor/results/test_qual/CROP/
### bash experiments/cal_metric.sh is /root/Proj/DeepFaceDrawing-Jittor/results/test_qual/CROP/
