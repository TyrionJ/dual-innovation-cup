def calculate_iou(bbox_a, bbox_b):
    a_xmin, a_ymin, a_xmax, a_ymax = bbox_a[0], bbox_a[1], bbox_a[2], bbox_a[3]
    a_width = a_xmax - a_xmin
    a_height = a_ymax - a_ymin

    b_xmin, b_ymin, b_xmax, b_ymax = bbox_b[0], bbox_b[1], bbox_b[2], bbox_b[3]
    b_width = b_xmax - b_xmin
    b_height = b_ymax - b_ymin

    xmin = min(a_xmin, b_xmin)
    ymin = min(a_ymin, b_ymin)
    xmax = max(a_xmax, b_xmax)
    ymax = max(a_ymax, b_ymax)

    a_width_and = (a_width + b_width) - (xmax - xmin)
    a_height_and = (a_height + b_height) - (ymax - ymin)

    if a_width_and <= 0.0001 or a_height_and <= 0.0001:
        return 0

    area_and = a_width_and * a_height_and
    area_or = a_width * a_height + b_width * b_height
    iou = area_and / (area_or - area_and)

    return iou
