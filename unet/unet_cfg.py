from proj_cfg import unet_root

unet_data_folder = f'{unet_root}/data'

dir_img = f'{unet_data_folder}/imgs/'
dir_mask = f'{unet_data_folder}/masks/'
dir_output = f'{unet_data_folder}/output'

epoch_file = f'{unet_data_folder}/epoch.txt'
result_file = f'{unet_data_folder}/result.txt'
meta_file = f'{unet_data_folder}/meta_file.txt'

dir_checkpoint = f'{unet_root}/checkpoints'
best_model = f'{dir_checkpoint}/best.pth'
last_model = f'{dir_checkpoint}/last.pth'

n_channels = 3
n_classes = 1

