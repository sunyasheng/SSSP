jobname=$1

w_intermediate_result=True
is_video_gen=False

run_test () {
    jobname=$1
    config_path=$2
    load_path=$3
    gpu_ids=$4
    test_meta_path=$5
    w_intermediate_result=$6
    is_video_gen=$7
    # echo ${w_intermediate_result}

    cmd="python3.7 test.py \
        --name ${jobname} \
        --config_file $config_path \
        --num_node 1 --tensorboard \
        --gpu_ids ${gpu_ids} \
        --load_path ${load_path} \
        --meta_path ${test_meta_path} \
        --w_intermediate_result ${w_intermediate_result} \
        --is_video_gen ${is_video_gen}"
    echo $cmd
    $cmd
}


######################### w space ##############################
########## image ###########
if [[ ${jobname} == 'sketch_w_128_quanti' ]]; then
    config_path=configs/testing/sketch_128_quanti.yaml
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse/checkpoint/000060e_171999iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_triplet/checkpoint/000037e_108793iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_wD/checkpoint/000006e_20040iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip/checkpoint/000025e_73999iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000025e_89309iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab_clipsso/checkpoint/000027e_80163iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab_clipsso/checkpoint/000046e_131999iter.pth
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000131e_198527iter.pth
    
    # ablation w/o region-aware rendering
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
    # ablation w/o triplane reverse
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_anchor_patchsampling_rgb/checkpoint/000017e_57999iter.pth
    gpu_ids=0,
    # w_intermediate_result=False
    w_intermediate_result=True
    test_meta_path=configs/flists/test_finegrained_sketch_flists_quanti.txt
    # test_meta_path=configs/flists/test_finegrained_sketch_flists_addition.txt
    run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
fi


if [[ ${jobname} == 'sketch_w_128_quanti_flip' ]]; then
    config_path=configs/testing/sketch_128_quanti_flip.yaml
    
    # ablation w/o region-aware rendering
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_anchor_patchsampling_rgb/checkpoint/000017e_57999iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000131e_198527iter.pth
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth

    gpu_ids=0,
    w_intermediate_result=True
    test_meta_path=configs/flists/test_finegrained_sketch_flists_quanti.txt
    run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
fi


############## video ##############
if [[ ${jobname} == 'sketch_w_128_quanti_video' ]]; then
    config_path=configs/testing/sketch_128_quanti_bs1.yaml
    
    # ablation full model
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000131e_198527iter.pth
    # ablation w/o region-aware rendering
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
    # ablation w/o triplane reverse
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_anchor_patchsampling_rgb/checkpoint/000017e_57999iter.pth
    gpu_ids=0,
    
    # w_intermediate_result=False
    w_intermediate_result=True
    is_video_gen=True
    # test_meta_path=configs/flists/test_finegrained_sketch_flists_quanti_ablation.txt
    test_meta_path=configs/flists/test_finegrained_sketch_flists_addition.txt
    run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
fi

if [[ ${jobname} == 'sketch_w_128_quanti_flip_video' ]]; then
    config_path=configs/testing/sketch_128_quanti_flip_bs1.yaml
    
    # ablation w/o region-aware rendering
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_anchor_patchsampling_rgb/checkpoint/000017e_57999iter.pth
    # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000131e_198527iter.pth
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth

    gpu_ids=0,
    w_intermediate_result=True
    is_video_gen=True
    test_meta_path=configs/flists/test_finegrained_sketch_flists_quanti_ablation.txt
    run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
fi

######################## w space #############################


######################## wp space #############################
######## image #########
if [[ ${jobname} == 'sketch_wp_128_quanti_flip' ]]; then
    config_path=configs/testing/sketch_wp_128_quanti_flip.yaml
    
    # ablation wp
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_wp_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000024e_85874iter.pth

    gpu_ids=1,
    w_intermediate_result=True
    test_meta_path=configs/flists/test_finegrained_sketch_flists_quanti.txt
    run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
fi



if [[ ${jobname} == 'sketch_wp_128_quanti' ]]; then
    config_path=configs/testing/sketch_wp_128_quanti.yaml
    
    # ablation wp
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_wp_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000024e_85874iter.pth

    gpu_ids=1,
    w_intermediate_result=True
    test_meta_path=configs/flists/test_finegrained_sketch_flists_quanti.txt
    run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
fi

######## video #########
if [[ ${jobname} == 'sketch_wp_128_quanti_flip_video' ]]; then
    config_path=configs/testing/sketch_wp_128_quanti_flip_bs1.yaml
    
    # ablation wp
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_wp_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000024e_85874iter.pth

    gpu_ids=0,
    w_intermediate_result=True
    is_video_gen=True
    test_meta_path=configs/flists/test_finegrained_sketch_flists_quanti_ablation.txt
    run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
fi



if [[ ${jobname} == 'sketch_wp_128_quanti_video' ]]; then
    config_path=configs/testing/sketch_wp_128_quanti_bs1.yaml
    
    # ablation wp
    load_path=OUTPUT/fs_ffhq_resnet_dualcontr_wp_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000024e_85874iter.pth

    gpu_ids=0,
    w_intermediate_result=True
    is_video_gen=True
    test_meta_path=configs/flists/test_finegrained_sketch_flists_quanti_ablation.txt
    run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
fi
######################## wp space #############################


# if [[ ${jobname} == 'sketch_w_128_quali' ]]; then
#     config_path=configs/testing/sketch_128_quali.yaml
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse/checkpoint/000060e_171999iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_triplet/checkpoint/000037e_108793iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_wD/checkpoint/000006e_20040iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip/checkpoint/000025e_73999iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling/checkpoint/000023e_82439iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000025e_89309iter.pth
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab_clipsso/checkpoint/000027e_80163iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/test_finegrained_sketch_flists_quali.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result} ${is_video_gen}
# fi


# if [[ $jobname == 'sketch_128' ]]; then
#     config_path=configs/testing/sketch_128.yaml
#     # load_path=OUTPUT/DATA_sketch_dt_ffhq_resnet_contr/checkpoint/000080e_233999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_dualcontr/checkpoint/000007e_87999iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/test_meta_flists.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi



# if [[ $jobname == 'psedu_sketch_128_z' ]]; then
#     config_path=configs/testing/psedu_sketch_128_z.yaml
#     # load_path=OUTPUT/DATA_sketch_dt_ffhq_resnet_contr/checkpoint/000080e_233999iter.pth
#     # load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_dualcontr/checkpoint/000007e_87999iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/psedu_test_meta_flists_same.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi


# if [[ $jobname == 'psedu_sketch_128_w' ]]; then
#     config_path=configs/testing/psedu_sketch_128_w.yaml
#     # load_path=OUTPUT/ffhq_celeba_resnet_dualcontr_w/checkpoint/000015e_169999iter.pth
#     load_path=OUTPUT/ffhq_resnet_dualcontr_w_triplane_reverse/checkpoint/000099e_289999iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/psedu_test_meta_flists.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi


# if [[ ${jobname} == 'sketch_128_quali_z' ]]; then
#     config_path=configs/testing/sketch_128_quali_z.yaml
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling/checkpoint/000023e_82439iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/test_finegrained_sketch_flists_quali.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result}
# fi

# if [[ ${jobname} == 'sketch_w_freehand' ]]; then
#     config_path=configs/testing/sketch_128_freehand.yaml
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000025e_89309iter.pth
#     gpu_ids=0,
#     test_meta_path=/root/dataset/FFHQ/test_freehand/2nd_stg
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result}
# fi



# if [[ ${jobname} == 'sketch_w_freehand_edited' ]]; then
#     config_path=configs/testing/sketch_128_freehand.yaml
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000025e_89309iter.pth
#     gpu_ids=0,
#     # test_meta_path=/root/dataset/FFHQ/test_freehand_edited/
#     test_meta_path=/root/dataset/FFHQ/edited_freehand_sketch_wo_alpha/
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result}
# fi


# if [[ ${jobname} == 'sketch_wp_128_quali' ]]; then
#     config_path=configs/testing/sketch_wp_128_quali.yaml
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_wp_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000024e_85874iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/test_finegrained_sketch_flists_quali.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path} ${w_intermediate_result}
# fi



## for debug, not for test
# if [[ ${jobname} == 'sketch128_patch_sampling' ]]; then
#     config_path=configs/testing/sketch_128_patch_sampling.yaml
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse/checkpoint/000060e_171999iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_triplet/checkpoint/000037e_108793iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_wD/checkpoint/000006e_20040iter.pth
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_clip/checkpoint/000025e_73999iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/test_finegrained_sketch_flists.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi


# ### memory overflow, finetune means add another decoder to train
# if [[ ${jobname} == 'sketch128_finetune_color' ]]; then
#     config_path=configs/testing/sketch_128_finetune_color.yaml
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse/checkpoint/000060e_171999iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor/checkpoint/000046e_134560iter.pth
#     # load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_triplet/checkpoint/000037e_108793iter.pth
#     load_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_wD/checkpoint/000006e_20040iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/test_finegrained_sketch_flists.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi


# ### varitex settings
# load_path=OUTPUT/ffhq_celeba_resnet_dualcontr_varitex_w/checkpoint/000010e_119999iter.pth
# if [[ $jobname == 'psedu_sketch_128_varitex_w_puregeo' ]]; then
#     config_path=configs/testing/psedu_sketch_128_varitex_w_puregeo.yaml
#     gpu_ids=0,
#     test_meta_path=configs/flists/psedu_test_meta_flists.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi


# if [[ $jobname == 'psedu_sketch_128_varitex_w_puretex' ]]; then
#     config_path=configs/testing/psedu_sketch_128_varitex_w_puretex.yaml
#     gpu_ids=0,
#     test_meta_path=configs/flists/psedu_test_meta_flists.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi



# if [[ $jobname == 'psedu_sketch_128_varitex_w' ]]; then
#     config_path=configs/testing/psedu_sketch_128_varitex_w.yaml
#     gpu_ids=0,
#     test_meta_path=configs/flists/psedu_test_meta_flists.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi

# if [[ $jobname == 'sketch_64' ]]; then
#     config_path=configs/testing/sketch_64.yaml
#     # load_path=OUTPUT/DATA_sketch_dt_ffhq_resnet_contr/checkpoint/000080e_233999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     gpu_ids=0,
#     test_meta_path=configs/flists/test_meta_flists.txt
#     run_test ${jobname} ${config_path} ${load_path} ${gpu_ids} ${test_meta_path}
# fi
