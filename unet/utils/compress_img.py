import os

from PIL import Image


def get_size(file):
    size = os.path.getsize(file)
    return size / 1024


def resize_image(infile, outfile='', x_s=1024):
    im = Image.open(infile)
    x, y = im.size
    y_s = int(y * x_s / x)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    out.save(outfile)


if __name__ == '__main__':
    in_folder = '../data/imgs'
    out_folder = '../data/imgs_c'
    m_in_folder = '../data/masks'
    m_out_folder = '../data/masks_c'
    imgs = os.listdir(in_folder)
    for img in imgs:
        print(f'Processing {img} ...')
        resize_image(f'{in_folder}/{img}', f'{out_folder}/{img}')

    imgs = os.listdir(m_in_folder)
    for img in imgs:
        print(f'Processing {img} ...')
        resize_image(f'{m_in_folder}/{img}', f'{m_out_folder}/{img}')
