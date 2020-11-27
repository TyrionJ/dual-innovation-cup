from proj_cfg import unet_root

mode = 'all'  # 'vertical'

unet_data_folder = f'{unet_root}/data'

dir_img = f'{unet_data_folder}/imgs/'
dir_mask = f'{unet_data_folder}/masks/'
dir_output = f'{unet_data_folder}/output'
dir_output_vertical = f'{unet_data_folder}/output_vertical'
dir_output_all = f'{unet_data_folder}/output_all'

epoch_file = f'{unet_data_folder}/epoch.txt'
result_file = f'{unet_data_folder}/result.txt'
meta_file = f'{unet_data_folder}/meta_file.json'
dice_file = f'{unet_data_folder}/dice.json'

dir_checkpoint = f'{unet_root}/checkpoints'
all_best_model = f'{dir_checkpoint}/all_best.pth'
all_last_model = f'{dir_checkpoint}/all_last.pth'
vertical_best_model = f'{dir_checkpoint}/vertical_best.pth'
vertical_last_model = f'{dir_checkpoint}/vertical_last.pth'

n_channels = 3
n_classes = 1
