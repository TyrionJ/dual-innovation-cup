import os
from PIL import Image, ImageDraw

from proj_cfg import ori_images_folder, marked_folder
from proj_utils.xml_parser import get_bounds


def draw_bnd_img(img_name):
    os.makedirs(marked_folder, exist_ok=True)

    img = Image.open(f'{ori_images_folder}/{img_name}')
    draw = ImageDraw.Draw(img)
    bnds = get_bounds(img_name.replace('JPG', 'xml'))
    for bnd in bnds:
        draw.rectangle(bnd.get_pos(), outline='red', width=10)
    img.save(f'{marked_folder}/{img_name}')
    del draw


if __name__ == '__main__':
    for imgs in os.listdir(ori_images_folder):
        print(f'Drawing {imgs} ...', end='')
        draw_bnd_img(imgs)
        print(f'\rDrawing {imgs} [DONE]')
