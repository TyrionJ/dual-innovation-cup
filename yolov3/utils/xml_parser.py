import xml.etree.ElementTree as ET
from typing import List

from proj_cfg import bnd_box_folder


class BoundInfo:
    def __init__(self, item: ET.Element=None):
        if item is not None:
            bndbox = item.find('bndbox')
            self.class_id = str(int(item.find('name').text)-1)
            self.x_min = int(bndbox.find('xmin').text)
            self.y_min = int(bndbox.find('ymin').text)
            self.x_max = int(bndbox.find('xmax').text)
            self.y_max = int(bndbox.find('ymax').text)

    def get_pos(self):
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def get_pos2(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    def to_dict(self):
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "class_id": self.class_id
        }


def has_label(xml_file):
    bound_file = open(f'{bnd_box_folder}/{xml_file}')
    tree = ET.parse(bound_file)
    root = tree.getroot()

    return root.find('labeled').text == 'true'


def get_bounds(bound_file) -> List[BoundInfo]:
    bound_file = open(f'{bnd_box_folder}/{bound_file}')
    tree = ET.parse(bound_file)
    root = tree.getroot()

    bnds = []
    for item in root.iter('item'):
        bnds.append(BoundInfo(item))

    return bnds


def get_img_size(image_id):
    in_file = open(f'{bnd_box_folder}/{image_id}.xml')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')

    w = int(size.find('width').text)
    h = int(size.find('height').text)
    d = int(size.find('depth').text)

    return [w, h, d]
