import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from unet.model import UNet
from unet.unet_cfg import dir_img, dir_output,\
    all_best_model, vertical_best_model,\
    dir_output_vertical, dir_output_all
from unet.utils.dataset import BasicDataset


def predict_img(net_all,
                net_vertical,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net_all.eval()
    net_vertical.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output_all = net_all(img)
        output_vertical = net_vertical(img)

        probs_all = torch.sigmoid(output_all).squeeze(0)
        probs_vertical = torch.sigmoid(output_vertical).squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs_all = tf(probs_all.cpu())
        full_mask_all = probs_all.squeeze().cpu().numpy()

        probs_vertical = tf(probs_vertical.cpu())
        full_mask_vertical = probs_vertical.squeeze().cpu().numpy()
    # tm = Image.open(f'{dir_mask}/{img_id}.png').convert('1')

    mask_all = full_mask_all > out_threshold
    mask_vertical = full_mask_vertical > out_threshold
    combined_mask = np.logical_or(mask_all, mask_vertical)

    return mask_all, mask_vertical, combined_mask


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', '-i', metavar='INPUT', nargs='+', default=dir_img,
                        help='filenames of input images')
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":

    args = get_args()
    in_files = os.listdir(args.source)

    net_all = UNet(n_channels=3, n_classes=1)
    net_vertical = UNet(n_channels=3, n_classes=1)

    print(f'Loading model {all_best_model}, {vertical_best_model}')

    device = torch.device('cuda' if False and torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    net_all.to(device=device)
    net_vertical.to(device=device)
    net_all.load_state_dict(torch.load(all_best_model, map_location=device))
    net_vertical.load_state_dict(torch.load(vertical_best_model, map_location=device))
    print("Model loaded !")

    os.makedirs(dir_output_all, exist_ok=True)
    os.makedirs(dir_output_vertical, exist_ok=True)
    os.makedirs(dir_output, exist_ok=True)

    for i, file in enumerate(in_files):
        print(f"\nPredicting image {file} ...")

        image = Image.open(f'{args.source}/{file}')
        _all, _vertical, _combined = predict_img(net_all=net_all,
                                                 net_vertical=net_vertical,
                                                 full_img=image,
                                                 scale_factor=args.scale,
                                                 out_threshold=args.mask_threshold,
                                                 device=device)

        if not args.no_save:
            result_all = mask_to_image(_all)
            result_all = result_all.resize(image.size, Image.ANTIALIAS)
            result_all.save(f'{dir_output_all}/{file}')

            result_vertical = mask_to_image(_vertical)
            result_vertical = result_vertical.resize(image.size, Image.ANTIALIAS)
            result_vertical.save(f'{dir_output_vertical}/{file}')

            result = mask_to_image(_combined)
            result = result.resize(image.size, Image.ANTIALIAS)
            result.save(f'{dir_output}/{file}')

            print("combined mask saved to {}".format(dir_output))
