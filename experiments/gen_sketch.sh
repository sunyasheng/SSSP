# version='torch'
version=$1

input_img_root=/home/ssd2/sunyasheng/Dataset/FFHQ
output_sketch_root=/home/ssd2/sunyasheng/Dataset/FFHQ_edge
# echo ${version}


if [[ ${version} == 'torch' ]]; then
        model_ckpt_path=/home/ssd3/sunyasheng/Proj/model_gan.t7
        # input_img_root=/home/vis/sunyasheng/Dataset/FFHQ_sample
        # output_sketch_root=/home/vis/sunyasheng/Dataset/FFHQ_sample_edge
        python dataset_preprocessing/generate_sketch_data.py \
                --model_ckpt_path ${model_ckpt_path} \
                --input_img_root ${input_img_root} \
                --output_sketch_root ${output_sketch_root}
fi


if [[ ${version} == 'pytorch' ]]; then
        model_ckpt_path=OUTPUT/pretrained/model_gan.pth
        input_img_root=/home/vis/sunyasheng/Dataset/FFHQ_sample
        output_sketch_root=/home/vis/sunyasheng/Dataset/FFHQ_sample_edge
        python dataset_preprocessing/generate_sketch_data_pytorch.py \
                --model_ckpt_path ${model_ckpt_path} \
                --input_img_root ${input_img_root} \
                --output_sketch_root ${output_sketch_root}
fi
