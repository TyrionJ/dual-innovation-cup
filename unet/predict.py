import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet.model import UNet
from unet.unet_cfg import best_model, dir_img, dir_output
from unet.utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', default=best_model,
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--source', '-i', metavar='INPUT', nargs='+', default=dir_img,
                        help='filenames of input images')
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', default=dir_output,
                        help='Filenames of ouput images')
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

    net = UNet(n_channels=3, n_classes=1)

    print("Loading model {}".format(args.model))

    device = torch.device('cuda' if False and torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    print("Model loaded !")

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for i, file in enumerate(in_files):
        print(f"\nPredicting image {file} ...")

        image = Image.open(f'{args.source}/{file}')
        mask = predict_img(net=net,
                           full_img=image,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out = f'{args.output}/{file}'
            result = mask_to_image(mask)
            result.save(out)

            print("Mask saved to {}".format(out))
