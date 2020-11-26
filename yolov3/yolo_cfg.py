from proj_cfg import yolo_root

yolo_classes = ['bomb']

yolo_data_folder = f'{yolo_root}/data'
yolo_images = f'{yolo_data_folder}/images'
yolo_txt_folder = f'{yolo_data_folder}/txt'
yolo_label_folder = f'{yolo_data_folder}/labels'
yolo_pred_output = f'{yolo_data_folder}/output'
yolo_result_file = f'{yolo_data_folder}/results.txt'

yolo_data = f'{yolo_root}/cfg/yolo.data'
yolo_names = f'{yolo_root}/cfg/yolo.names'
yolo_cfg = f'{yolo_root}/cfg/yolov3-spp.cfg'
yolo_cfg_small = f'{yolo_root}/cfg/yolov3-spp-small.cfg'
yolo_weights_folder = f'{yolo_root}/weights'

yolo_weights = f'{yolo_weights_folder}/yolov3-spp-ultralytics.pt'
yolo_best = f'{yolo_weights_folder}/best.pt'
yolo_last = f'{yolo_weights_folder}/last.pt'

img_size = 512
