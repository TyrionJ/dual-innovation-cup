from PIL import Image, ImageDraw

from proj_cfg import ori_images_folder
from proj_utils.xml_parser import get_bounds, BoundInfo


def draw_bnd_img(img_name, ann_file=None):
    img = Image.open(f'{ori_images_folder}/{img_name}')
    draw = ImageDraw.Draw(img)
    bnds = BoundInfo.load_bnds(ann_file) if ann_file is not None else get_bounds(img_name.replace('JPG', 'xml'))
    for bnd in bnds:
        draw.rectangle(bnd.get_pos(), outline='red', width=3)
    img.save(f'/{img_name}')
    del draw


if __name__ == '__main__':
    draw_bnd_img('014.JPG')
