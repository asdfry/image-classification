import os
import cv2
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm


def img_to_numpy(img_path, res):
    img = cv2.imread(img_path)  # Read image
    upscaled_img = cv2.resize(  # Upsacle image
        img,
        dsize=(resolutions[res][0], resolutions[res][1]),
        interpolation=cv2.INTER_LINEAR,
    )
    img_array = np.asarray(upscaled_img)  # Image to array
    # fn.crop_mirror_normalize(output_layout="CHW")로 해결 가능하므로 아래 코드는 주석 처리
    # img_array = np.moveaxis(img_array, -1, 0)  # Channel first
    return img_array


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str)
parser.add_argument("-r", "--res", type=str)
parser.add_argument("-hf", "--half", action="store_true")
parser.add_argument("-qt", "--quarter", action="store_true")
args = parser.parse_args()

black_list = []
resolutions = {
    "sd": (854, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
    "qhd": (2560, 1440),
    "uhd": (3840, 2160),
}

if args.quarter:
    npy_dir = f"{args.dir}-quarter-numpy"
    div = 4
elif args.half:
    npy_dir = f"{args.dir}-half-numpy"
    div = 2
else:
    npy_dir = f"{args.dir}-numpy"
    div = 1

npy_dir = f"{npy_dir}-{args.res}"
labels = {}

tv = "train"
os.makedirs(f"{npy_dir}/{tv}", exist_ok=True)
img_paths = glob(f"{args.dir}/{tv}/**/*.JPEG")

for img_path in tqdm(img_paths[: len(img_paths) // div]):
    if img_path in black_list:
        continue

    img_array = img_to_numpy(img_path, args.res)

    # Skip grayscale image
    if len(img_array.shape) == 2:
        black_list.append(img_path)
        continue

    file = img_path.split("/")[-1]
    file = file.replace("JPEG", "image.npy")
    np.save(f"{npy_dir}/{tv}/{file}", img_array)

    # Label
    label = img_path.split("/")[-2]
    if not label in labels:
        labels[label] = len(labels)
    file = file.replace("image.npy", "label.npy")
    np.save(f"{npy_dir}/{tv}/{file}", np.array([labels[label]]))

print(f"[{args.res}] Done converting to numpy ({tv})")

tv = "val"
os.makedirs(f"{npy_dir}/{tv}", exist_ok=True)
img_paths = glob(f"{args.dir}/{tv}/**/*.JPEG")

for img_path in tqdm(img_paths):
    label = img_path.split("/")[-2]

    if img_path in black_list:
        continue
    elif not label in labels:
        continue

    img_array = img_to_numpy(img_path, args.res)

    # Skip grayscale image
    if len(img_array.shape) == 2:
        black_list.append(img_path)
        continue

    file = img_path.split("/")[-1]
    file = file.replace("JPEG", "image.npy")
    np.save(f"{npy_dir}/{tv}/{file}", img_array)

    # Label
    file = file.replace("image.npy", "label.npy")
    np.save(f"{npy_dir}/{tv}/{file}", np.array([labels[label]]))

print(f"[{args.res}] Done converting to numpy ({tv})")
