__all__ = ['ori_images_folder',
           'mask_folder',
           'bnd_box_folder',
           'yolo_root',
           'pspnet_root',
           'classes',
           'unet_root']

classes = ['bomb']

root = 'D:/Program/Python/dual-innovation-cup'

data_folder = f'{root}/data'

ori_images_folder = f'{data_folder}/01-自爆缺陷原图'
mask_folder = f'{data_folder}/02-基于原图的标准掩模图'
bnd_box_folder = f'{data_folder}/03-自爆绝缘子BoundingBox标签'

yolo_root = f'{root}/yolov3'
pspnet_root = f'{root}/pspnet_lib'
unet_root = f'{root}/unet'
