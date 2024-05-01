import shutil
import sys
import os
import argparse
import glob
import random


def img2sketch_root(img_root):
    sketch_root = img_root.replace('CROP', 'CROP_sketch_sota')
    return sketch_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_root', type=str, default='')
    parser.add_argument('--test_image_root', type=str, default='')
    args = parser.parse_args()

    train_sketch_root = img2sketch_root(args.train_image_root)
    test_sketch_root = img2sketch_root(args.test_image_root)

    os.makedirs(args.test_image_root, exist_ok=True)
    os.makedirs(test_sketch_root, exist_ok=True)
    
    train_sketch_paths = glob.glob(os.path.join(train_sketch_root, '*.png'))
    random.shuffle(train_sketch_paths)
    
    chosen_train_sketch_paths = train_sketch_paths[:2500]
    chosen_train_image_paths = [p.replace(train_sketch_root, args.train_image_root) for p in chosen_train_sketch_paths]

    chosen_test_sketch_paths = [p.replace(train_sketch_root, test_sketch_root) for p in chosen_train_sketch_paths]
    chosen_test_image_paths = [p.replace(args.train_image_root, args.test_image_root) for p in chosen_train_image_paths]
    
    # import pdb; pdb.set_trace();
    for i, (a,b) in enumerate(zip(chosen_train_image_paths, chosen_test_image_paths)):
        print(i, a,b)
        shutil.move(a,os.path.dirname(b))
        # break
    
    for i, (a,b) in enumerate(zip(chosen_train_sketch_paths, chosen_test_sketch_paths)):
        print(i, a,b)
        shutil.move(a,os.path.dirname(b))
        # break

    import pdb; pdb.set_trace();


if __name__ == '__main__':
    main()