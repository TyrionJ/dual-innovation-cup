import os
import random

import numpy as np
import matplotlib.image as Image
import PIL

from proj_cfg import ori_images_folder, mask_folder
from proj_utils.xml_parser import has_label, BoundInfo, get_bounds

from yolov3.yolo_cfg import yolo_images, yolo_txt_folder, yolo_label_folder, yolo_data_folder, yolo_data, yolo_names, \
    yolo_classes


def generate_images():
    if not os.path.exists(yolo_images):
        os.makedirs(yolo_images, exist_ok=True)

    files = os.listdir(ori_images_folder)
    for file in files:
        if has_label(f'{file[:-4]}.xml'):
            print(f'Generating {file} ...')
            ori_img = Image.imread(f'{ori_images_folder}/{file}')
            mask_img = Image.imread(f'{mask_folder}/{file[:-4]}.png')
            out_img = np.multiply(ori_img, mask_img[:, :, 0:3])
            Image.imsave(f'{yolo_images}/{file}', out_img / 255.)
            print(f' Saved {file} to {yolo_images}/{file}')
        else:
            print(f' Image {file} has no bbox, pass')


def make_txt():
    print('Generating train, test and validate set ...', end='')
    if not os.path.exists(yolo_txt_folder):
        os.mkdir(yolo_txt_folder)

    train_val_percent = 0.1
    train_percent = 0.9
    total_imgs = os.listdir(yolo_images)

    num = len(total_imgs)
    n_list = range(num)
    tv = int(num * train_val_percent)
    tr = int(tv * train_percent)
    train_val = random.sample(n_list, tv)
    train = random.sample(train_val, tr)

    f_train_val = open(f'{yolo_txt_folder}/train_val.txt', 'w')
    f_test = open(f'{yolo_txt_folder}/test.txt', 'w')
    f_train = open(f'{yolo_txt_folder}/train.txt', 'w')
    f_val = open(f'{yolo_txt_folder}/val.txt', 'w')

    for i in n_list:
        name = total_imgs[i][:-4] + '\n'
        if i in train_val:
            f_train_val.write(name)
            if i in train:
                f_test.write(name)
            else:
                f_val.write(name)
        else:
            f_train.write(name)

    f_train_val.close()
    f_train.close()
    f_val.close()
    f_test.close()

    print('\rGenerating train, test and validate set [DONE]')


def make_label():
    print('Generating labels ...', end='')
    if not os.path.exists(yolo_label_folder):
        os.makedirs(yolo_label_folder)

    sets = ['train', 'test', 'val']

    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return [x, y, w, h]

    def convert_annotation(image_id):
        out_file = open(f'{yolo_label_folder}/{image_id}.txt', 'w')

        img = PIL.Image.open(f'{yolo_images}/{image_id}.JPG')
        w, h = img.size

        bnds = get_bounds(image_id + '.xml')
        for bnd in bnds:
            bb = convert((w, h), bnd.get_pos2())
            out_file.write(bnd.class_id + " " + " ".join([str(a) for a in bb]) + '\n')

    for set in sets:
        image_ids = open(f'{yolo_txt_folder}/{set}.txt').read().strip().split()
        list_file = open(f'{yolo_data_folder}/{set}.txt', 'w')
        for image_id in image_ids:
            list_file.write(f'{yolo_images}/{image_id}.JPG\n')
            convert_annotation(image_id)
        list_file.close()

    print('\rGenerating labels [DONE]')


def make_yolo_data():
    print('Generating yolo data ...', end='')

    f = open(yolo_data, 'w')
    f.write(f'classes={len(yolo_classes)}\n')
    f.write(f'train={yolo_data_folder}/train.txt\n')
    f.write(f'valid={yolo_data_folder}/val.txt\n')
    f.write(f'names={yolo_names}\n')
    f.write(f'backup={yolo_data_folder}/backup\n')

    print('\rGenerating yolo data [DONE]')


def make_yolo_names():
    print('Generating yolo names ...', end='')

    f = open(yolo_names, 'w')
    f.write(yolo_classes[0])
    for i in range(1, len(yolo_classes)):
        f.write(f'\n{yolo_classes[i]}')

    print('\rGenerating yolo data [DONE]')


if __name__ == '__main__':
    generate_images()
    make_txt()
    make_label()
    make_yolo_data()
    make_yolo_names()
