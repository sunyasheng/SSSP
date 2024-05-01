from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.serialization import load_lua
import os
import cv2
import numpy as np
import argparse



"""
NOTE!: Must have torch==0.4.1 and torchvision==0.2.1
The sketch simplification model (sketch_gan.t7) from Simo Serra et al. can be downloaded from their official implementation: 
    https://github.com/bobbens/sketch_simplification
"""


parser = argparse.ArgumentParser()
parser.add_argument('--model_ckpt_path', type=str, default='')
parser.add_argument('--input_img_root', type=str, default='')
parser.add_argument('--output_sketch_root', type=str, default='')
args = parser.parse_args()


def sobel(img):
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    return cv2.bitwise_or(opImgx, opImgy)


def sketch(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    invImg = 255 - frame
    edgImg0 = sobel(frame)
    edgImg1 = sobel(invImg)
    edgImg = cv2.addWeighted(edgImg0, 0.75, edgImg1, 0.75, 0)
    opImg = 255 - edgImg
    return opImg


def get_sketch_image(image_path):
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    sketch_image = sketch(original)
    return sketch_image[:, :, np.newaxis]


use_cuda = True

cache = load_lua(args.model_ckpt_path)
model = cache.model
immean = cache.mean
imstd = cache.std
model.evaluate()

data_path = args.input_img_root
images = [os.path.join(data_path, f) for f in os.listdir(data_path)]
images = [p for p in images if '.png' in p or '.jpg' in p]
output_dir = args.output_sketch_root
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for idx, image_path in enumerate(images):
    if idx % 50 == 0:
        print("{} out of {}".format(idx, len(images)))
    data = get_sketch_image(image_path)
    data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
    if use_cuda:
        pred = model.cuda().forward(data.cuda()).float()
    else:
        pred = model.forward(data)
    save_image(pred[0], os.path.join(output_dir, "{}_edges.jpg".format(image_path.split("/")[-1].split('.')[0])))