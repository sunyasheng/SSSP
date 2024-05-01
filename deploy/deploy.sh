jobname=$1


run_mar(){
    checkpoint_path=$1
    extra_files=$2
    torch-model-archiver --model-name sketch2nerf \
            --version 1.0 --serialized-file ${checkpoint_path} \
            --handler deploy/handler.py --extra-files ${extra_files}
    mkdir -p model_store
    mv sketch2nerf.mar model_store
}


run_server(){
    torchserve --start --model-store model_store --models sketch2nerf=sketch2nerf.mar --ts-config deploy/config.properties --ncs
}


if [[ ${jobname} == 'mar' ]]; then
    checkpoint_path=OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000025e_89309iter.pth
    extra_files="./configs/testing/sketch_128_quali.yaml"
    run_mar ${checkpoint_path} ${extra_files}
fi


if [[ ${jobname} == 'server' ]]; then
    run_server
fi
