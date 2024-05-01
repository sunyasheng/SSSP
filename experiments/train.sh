jobname=$1
export http_proxy='http://agent.baidu.com:8188'
export https_proxy='http://agent.baidu.com:8188'


run_train () {
    jobname=$1
    config_path=$2
    load_path=$3
    gpu_ids=$4

    cmd="python3.8 train.py \
        --name ${jobname} \
        --config_file $config_path \
        --num_node 1 --tensorboard \
        --gpu_ids ${gpu_ids} \
        --load_path ${load_path}"
    echo $cmd
    $cmd
}

####### 1. z + camera == wp, When camera changes, the wp changes according to experiments. Therefore, we use wp setting.
####### 2. fs means fine-grained sketch, another extracted sketch datasets by tog2021
####### 3. w_triplane_reverse means camera is mirrored
####### 4. Considering data distribution of eg3d, only trained on FFHQ dataset currently


##### main setting
if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab' ]]; then
    config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab.yaml
    # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
    # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000025e_89309iter.pth
    gpu_ids=0,1
    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi


##### w/o triplane-reverse ## color get some problems, very wired
if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_anchor_patchsampling_lab' ]]; then
    config_path=configs/training/fs_ffhq_resnet_dualcontr_w_anchor_patchsampling_lab.yaml
    # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
    # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
    load_path=OUTPUT/000025e_89309iter.pth
    gpu_ids=0,
    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi


##### w/o triplane-reverse ## color get some problems, very wired
if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_anchor_patchsampling_rgb' ]]; then
    config_path=configs/training/fs_ffhq_resnet_dualcontr_w_anchor_patchsampling_rgb.yaml
    # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
    # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
    load_path=OUTPUT/000025e_89309iter.pth
    gpu_ids=0,
    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi



### ablation: w/o region-aware
if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor' ]]; then
    config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor.yaml
    # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
    # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse/checkpoint/000060e_171999iter.pth
    gpu_ids=0,
    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi

### ablation: wp
if [[ $jobname == 'fs_ffhq_resnet_dualcontr_wp_triplane_reverse_w_anchor_patchsampling_lab' ]]; then
    config_path=configs/training/fs_ffhq_resnet_dualcontr_wp_triplane_reverse_w_anchor_patchsampling_lab.yaml
    # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
    # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling/checkpoint/000076e_264494iter.pth
    gpu_ids=0,
    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi



# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip/checkpoint/000041e_120245iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_zcam_triplane_reverse_w_anchor_patchsampling' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_zcam_triplane_reverse_w_anchor_patchsampling.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip/checkpoint/000041e_120245iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_arcface' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_arcface.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


###### currently, we do not add cosine similarity loss on the fc since our patch is 64*64
# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab_clipsso' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab_clipsso.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling/checkpoint/000076e_264494iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_rstb' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_rstb.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse/checkpoint/000060e_171999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/ffhq_resnet_dualcontr_w_triplane_reverse/000100e_291999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'ffhq_resnet_dualcontr_w_triplane_reverse' ]]; then
#     config_path=configs/training/ffhq_resnet_dualcontr_w_triplane_reverse.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/ffhq_resnet_dualcontr_w_triplane_reverse/000100e_291999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_edge' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_edge.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# ## In this setting, relax mapper to cover different color patterns, does not converges well, not a good idea to disentangle before wp
# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab_finetune_z2w' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab_finetune_z2w.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000019e_68699iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_wD' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_wD.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip/checkpoint/000020e_57999iter.pth
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_wD/checkpoint/000006e_20040iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi




# if [[ $jobname == 'fs_ffhq_resnet_cam_pose_zcam' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_cam_pose_zcam.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip/checkpoint/000041e_120245iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip_wD' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip_wD.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_triplet' ]]; then
#     config_path=configs/training/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_triplet.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000012e_35999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'ffhq_128_resnet_dualcontr_w_triplane_reverse' ]]; then
#     config_path=configs/training/ffhq_128_resnet_dualcontr_w_triplane_reverse.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/ffhq_resnet_dualcontr_w_triplane_reverse/000100e_291999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi




# if [[ $jobname == 'ffhq_128_resnet_dualcontr_w_triplane_reverse_edge_l1' ]]; then
#     config_path=configs/training/ffhq_128_resnet_dualcontr_w_triplane_reverse_edge_l1.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter
#     load_path=OUTPUT/ffhq_resnet_dualcontr_w_triplane_reverse/000100e_291999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi
