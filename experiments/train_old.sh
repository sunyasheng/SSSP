if [[ $jobname == 'ffhq_celeba_resnet_dualcontr_csft' ]]; then
    config_path=configs/training/ffhq_celeba_resnet_dualcontr_csft.yaml
    # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
    load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
    gpu_ids=0,
    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi


if [[ $jobname == "DATA_ffhq" ]]; then
    config_path=configs/training/DATA_ffhq.yaml
    # load_path=OUTPUT/vox1_lrw_flat_ref_mouthloss_netA_rfp_mixvq_halfmask_nofc_wsync_pc/checkpoint/000000e_49999iter.pth # good sync
    # load_path=OUTPUT/Pretrain/l1loss/000058e_255999iter.pth
    load_path=OUTPUT/DATA_ffhq/checkpoint/000012e_69999iter.pth
    gpu_ids=0,

    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi


if [[ $jobname == "DATA_ldmk_ffhq" ]]; then
    config_path=configs/training/DATA_ldmk_ffhq.yaml
    # load_path=OUTPUT/vox1_lrw_flat_ref_mouthloss_netA_rfp_mixvq_halfmask_nofc_wsync_pc/checkpoint/000000e_49999iter.pth # good sync
    # load_path=OUTPUT/Pretrain/l1loss/000058e_255999iter.pth
    load_path=OUTPUT/DATA_ldmk_ffhq/checkpoint/000003e_17487iter.pth
    gpu_ids=0,

    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi


if [[ $jobname == "DATA_ldmk_dt_ffhq" ]]; then
    config_path=configs/training/DATA_ldmk_dt_ffhq.yaml
    # load_path=OUTPUT/vox1_lrw_flat_ref_mouthloss_netA_rfp_mixvq_halfmask_nofc_wsync_pc/checkpoint/000000e_49999iter.pth # good sync
    # load_path=OUTPUT/Pretrain/l1loss/000058e_255999iter.pth
    load_path=OUTPUT/DATA_ldmk_ffhq/checkpoint/000003e_17487iter.pth
    gpu_ids=0,

    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi


if [[ $jobname == 'DATA_sketch_dt_ffhq' ]]; then
    config_path=configs/training/DATA_sketch_dt_ffhq.yaml
    load_path=OUTPUT/DATA_ldmk_ffhq/checkpoint/000003e_17487iter.pth
    gpu_ids=0,
    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi



if [[ $jobname == 'DATA_sketch_ffhq' ]]; then
    config_path=configs/training/DATA_sketch_ffhq.yaml
    load_path=OUTPUT/DATA_sketch_dt_ffhq/checkpoint/000015e_65999iter.pth
    gpu_ids=0,
    run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
fi




# if [[ $jobname == 'ffhq_celeba_resnet_dualcontr_z' ]]; then
#     config_path=configs/training/ffhq_celeba_resnet_dualcontr_z.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'ffhq_celeba_resnet_dualcontr_z' ]]; then
#     config_path=configs/training/ffhq_celeba_resnet_dualcontr_z.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'ffhq_resnet_dualcontr_z_triplane_reverse' ]]; then
#     config_path=configs/training/ffhq_resnet_dualcontr_z_triplane_reverse.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'ffhq_celeba_resnet_contr' ]]; then
#     config_path=configs/training/ffhq_celeba_resnet_contr.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'DATA_sketch_dt_ffhq_resnet' ]]; then
#     config_path=configs/training/DATA_sketch_dt_ffhq_resnet.yaml
#     load_path=OUTPUT/DATA_sketch_dt_ffhq/checkpoint/000015e_65999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'DATA_sketch_dt_ffhq_resnet_contr' ]]; then
#     config_path=configs/training/DATA_sketch_dt_ffhq_resnet_contr.yaml
#     load_path=OUTPUT/DATA_sketch_dt_ffhq/checkpoint/000015e_65999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'DATA_sketch_ffhq_resnet_contr' ]]; then
#     config_path=configs/training/DATA_sketch_ffhq_resnet_contr.yaml
#     load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'ffhq_celeba_resnet_contr_img_inversion' ]]; then
#     config_path=configs/training/ffhq_celeba_resnet_contr_img_inversion.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi

# if [[ $jobname == 'DATA_sketch_ffhq_resnet_edgecontr' ]]; then
#     config_path=configs/training/DATA_sketch_ffhq_resnet_edgecontr.yaml
#     load_path=OUTPUT/DATA_sketch_dt_ffhq_resnet/checkpoint/000090e_263999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



### TODO: check those two settings
# if [[ $jobname == 'DATA_sketch_dt_ffhq_resnet_sharecontr' ]]; then
#     config_path=configs/training/DATA_sketch_dt_ffhq_resnet_sharecontr.yaml
#     load_path=OUTPUT/DATA_sketch_dt_ffhq/checkpoint/000015e_65999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi


# if [[ $jobname == 'DATA_sketch_ffhq_resnet_edgesharecontr' ]]; then
#     config_path=configs/training/DATA_sketch_ffhq_resnet_edgesharecontr.yaml
#     load_path=OUTPUT/DATA_sketch_dt_ffhq_resnet/checkpoint/000090e_263999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'ffhq_celeba_resnet_dualcontr_w' ]]; then
#     config_path=configs/training/ffhq_celeba_resnet_dualcontr_w.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi



# if [[ $jobname == 'ffhq_celeba_resnet_dualcontr_varitex_w' ]]; then
#     config_path=configs/training/ffhq_celeba_resnet_dualcontr_varitex_w.yaml
#     # load_path=OUTPUT/DATA_sketch_ffhq_resnet_contr/checkpoint/000040e_117999iter.pth
#     load_path=OUTPUT/ffhq_celeba_resnet_contr/checkpoint/000009e_103999iter.pth
#     gpu_ids=0,
#     run_train ${jobname} ${config_path} ${load_path} ${gpu_ids}
# fi

