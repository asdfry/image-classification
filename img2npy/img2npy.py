import os
import cv2
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str)
parser.add_argument("-hf", "--half", action="store_true")
args = parser.parse_args()

img_dir = args.dir
black_list = []
resolutions = {
    "sd": (854, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
    "qhd": (2560, 1440),
    "uhd": (3840, 2160),
}

for res in resolutions.keys():
    h = 0
    w = 0
    img_cnt = 0
    npy_dir = f"{img_dir}-numpy-{res}" if not args.half else f"{img_dir}-half-numpy-{res}"

    for tv in ["train", "val"]:
        os.makedirs(f"{npy_dir}/{tv}", exist_ok=True)
        labels = []
        img_paths = glob(f"{img_dir}/{tv}/**/*.JPEG")

        for idx, img_path in enumerate(tqdm(img_paths)):
            if img_path in black_list:
                continue

            # Read image
            img = cv2.imread(img_path)

            # Upsacle image
            upscaled_img = cv2.resize(
                img,
                dsize=(resolutions[res][0], resolutions[res][1]),
                interpolation=cv2.INTER_LINEAR,
            )

            # Image to array
            img_array = np.asarray(upscaled_img)
            # fn.crop_mirror_normalize 인자로 output_layout="CHW"를 주면 되므로 아래 코드는 주석 처리함
            # img_array = np.moveaxis(img_array, -1, 0)
            # 흑백 이미지 or 가로 또는 세로가 224 미만인 이미지 스킵
            if len(img_array.shape) == 2 or img_array.shape[0] < 224 or img_array.shape[1] < 224:
                black_list.append(img_path)
                continue
            file = img_path.split("/")[-1]
            file = file.replace("JPEG", "image.npy")
            np.save(f"{npy_dir}/{tv}/{file}", img_array)

            # Label
            label = img_path.split("/")[-2]
            if not label in labels:
                labels.append(label)
            file = file.replace("image.npy", "label.npy")
            np.save(f"{npy_dir}/{tv}/{file}", np.array([len(labels)]))

            # For calculate resolution
            h += img_array.shape[0]
            w += img_array.shape[1]
            img_cnt += 1

            if args.half and img_cnt == len(img_paths) // 2:
                break

    print(f"{res}: {int(w/img_cnt)} x {int(h/img_cnt)}")
