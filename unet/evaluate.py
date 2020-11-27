import json
import os
import numpy as np
from PIL import Image

from unet.unet_cfg import dir_mask, dir_output, dice_file, dir_output_vertical, dir_output_all

'''
    @return [total_dice, amend_total_dice,
             all_dice, amend_all_dice, vertical_dice,
             amend_vertical_dice]
'''
def calculate_dice(img_id, amend=False):
    tm = Image.open(f'{dir_mask}/{img_id}.png').convert('1')
    tm = np.array(tm).flatten()

    t_pm = Image.open(f'{dir_output}/{img_id}.JPG').convert('1')
    a_pm = Image.open(f'{dir_output_all}/{img_id}.JPG').convert('1')
    v_pm = Image.open(f'{dir_output_vertical}/{img_id}.JPG').convert('1')

    t_pm = np.array(t_pm).flatten()
    a_pm = np.array(a_pm).flatten()
    v_pm = np.array(v_pm).flatten()

    dices = []
    for _p in [t_pm, a_pm, v_pm]:
        inter = np.argwhere(tm == _p).size
        outer = tm.size + _p.size
        dices.append(2 * inter / outer)

        t_idx = np.argwhere(tm > 0)
        inter = np.argwhere(_p[t_idx] > 0).shape[0]
        outer = t_idx.size
        dices.append(inter / outer)

    return dices


def write_dice(mask_id, dice):
    obj = json.load(open(dice_file)) if os.path.exists(dice_file) else {}
    obj[mask_id] = dice

    with open(dice_file, 'w') as f:
        json.dump(obj, f)


if __name__ == '__main__':
    imgs = os.listdir(dir_mask)
    for img in imgs:
        img_id = img[:-4]
        dice = calculate_dice(img_id)
        print(f'{img_id}: {dice}')
        write_dice(img_id, dice)
