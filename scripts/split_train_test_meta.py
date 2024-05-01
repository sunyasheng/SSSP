import shutil
import sys
import os
import argparse
import glob
import random

#### all sketch images are located in one root, split test/train by meta_path
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/root/dataset/FFHQ/train/CROP_sketch_sota/')
    parser.add_argument('--test_meta_path', type=str, default='configs/flists/test_finegrained_sketch_flists_quanti.txt')
    parser.add_argument('--train_meta_path', type=str, default='configs/flists/finegrained_sketch_flists.txt')
    args = parser.parse_args()

    sketch_paths = glob.glob(os.path.join(args.root, '*.png')) + \
                    glob.glob(os.path.join(args.root, '*.jpg'))

    all_names = [os.path.basename(path) for path in sketch_paths]
    test_names = open(args.test_meta_path, 'r').read().splitlines()
    train_names = list(set(all_names).difference(set(test_names)))
    train_names = [tn+'\n' for tn in train_names]
    
    with open(args.train_meta_path, 'w') as f:
        f.writelines(train_names)
    
    import pdb; pdb.set_trace();


if __name__ == '__main__':
    main()