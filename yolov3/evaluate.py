import os

from yolov3.utils.bbox_iou import calculate_iou
from yolov3.utils.xml_parser import get_bounds, BoundInfo
from yolov3.yolo_cfg import yolo_pred_output, yolo_images


def get_pred_bnds(img_id):
    bnds = []
    with open(f'{yolo_pred_output}/{img_id}.txt') as f:
        line = f.readline()
        while len(line) > 0:
            bnd = BoundInfo()
            line = line.split(',')
            bnd.class_id = line[0]
            bnd.x_min = int(line[1])
            bnd.y_min = int(line[2])
            bnd.x_max = int(line[3])
            bnd.y_max = int(line[4])

            bnds.append(bnd)
            line = f.readline()

    return bnds


def calculate_avg_iou(img_id):
    ori_bnds = get_bounds(img_id + '.xml')
    pred_bnds = get_pred_bnds(img_id)
    ori_bnds.sort(key=lambda bnd: bnd.x_min)
    pred_bnds.sort(key=lambda bnd: bnd.x_min)

    size = min(len(ori_bnds), len(pred_bnds))
    total_iou = 0
    max_iou = 0
    min_iou = 1
    for i in range(size):
        iou = calculate_iou(ori_bnds[i].get_pos(), pred_bnds[i].get_pos())
        if min_iou > iou:
            min_iou = iou
        if max_iou < iou:
            max_iou = iou
        total_iou += iou

    return round(total_iou / size, 6), round(min_iou, 6), round(max_iou, 6), size


if __name__ == '__main__':
    for img in os.listdir(yolo_images):
        print(img, calculate_avg_iou(img[:-4]))
