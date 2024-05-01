train_image_root=/root/dataset/FFHQ/train/CROP
test_image_root=/root/dataset/FFHQ/test_qual/CROP

python scripts/split_train_test.py \
        --train_image_root ${train_image_root} \
        --test_image_root ${test_image_root}