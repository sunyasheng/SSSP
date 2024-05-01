import os
import glob
import cv2


def cut_and_copy():
    cat_root = '/root/Proj/Sketch2Nerf/OUTPUT/test/psedu_sketch_128'
    cat_paths = glob.glob(os.path.join(cat_root, '*.png'))
    cat_paths += glob.glob(os.path.join(cat_root, '*.jpg'))
    print(len(cat_paths))
    print(cat_paths[:10])

    cut_root = '/root/Proj/Sketch2Nerf/OUTPUT/test/psedu_sketch_128_pred_render'

    for cat_path in cat_paths:
        cat_img = cv2.imread(cat_path)
        h = cat_img.shape[0]
        cut_path = cat_path.replace(cat_root, cut_root)
        cv2.imwrite(cut_path, cat_img[:, 2*h:3*h])


if __name__ == '__main__':
    cut_and_copy()
