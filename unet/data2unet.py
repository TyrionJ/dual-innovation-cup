import json
import os
from pathlib import Path

from PIL import Image

from unet.unet_cfg import meta_file, dir_img, dir_mask
from proj_cfg import ori_images_folder, mask_folder


def write_meta(file, w, h):
    p = Path(file)
    obj = json.load(open(meta_file)) if os.path.exists(meta_file) else {}
    obj[p.name] = [w, h]

    with open(meta_file, 'w') as f:
        json.dump(obj, f)


def load_meta(file):
    if not os.path.exists(file):
        return None
    p = Path(file)
    obj = json.load(meta_file)
    return obj[p.name]


def resize_image(in_file, out_file='', x_s=None):
    im = Image.open(in_file)
    x, y = im.size
    if x_s is not None:
        write_meta(in_file, x, y)
    y_s = int(y * x_s / x)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    out.save(out_file)


def compress(in_file, out_file, n_size):
    resize_image(in_file, out_file, n_size)


def recovery(in_file, out_file):
    w = load_meta(in_file)
    resize_image(in_file, out_file, w)


if __name__ == '__main__':
    if not os.path.exists(dir_img):
        os.mkdir(dir_img)
    if not os.path.exists(dir_mask):
        os.mkdir(dir_mask)

    print('Generating images ...')
    imgs = os.listdir(ori_images_folder)
    to_width = 1024
    for img in imgs:
        print(f' Processing {img} ...', end='')
        compress(f'{ori_images_folder}/{img}', f'{dir_img}/{img}', to_width)
        print(f'\r Processing {img} [DONE]')

    print('Generating masks ...')
    masks = os.listdir(mask_folder)
    for mask in masks:
        print(f' Processing {mask} ...', end='')
        compress(f'{mask_folder}/{mask}', f'{dir_mask}/{mask}', to_width)
        print(f'\r Processing {mask} [DONE]')
