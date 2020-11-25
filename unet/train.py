import argparse
import logging
import os
import sys
import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from unet.model import UNet

from torch.utils.tensorboard import SummaryWriter

from unet.unet_cfg import best_model, last_model,\
    epoch_file, result_file,\
    dir_checkpoint, dir_mask, dir_img, \
    n_channels, n_classes
from unet.utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


min_gpu = 0
torch.cuda.set_device(min_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = f'{min_gpu}'


def train_net(net,
              device,
              start_epoch=0,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_loss = 999
    for epoch in range(start_epoch, epochs):
        net.train()

        epoch_loss = 0
        batch_size = 0
        start_time = time.time()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                batch_size += 1
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])

        print('Start saving checkpoint ...', end='')

        avg_loss = round(epoch_loss / batch_size, 4)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), best_model)
        torch.save(net.state_dict(), last_model)

        f = open(epoch_file, 'w')
        f.write(str(epoch))
        f.close()

        time_pass = round(time.time() - start_time, 6)
        f2 = open(result_file, 'a')
        f2.write(f'{epoch + 1}/'
                 f'{epochs}\t'
                 f'{avg_loss}\t'
                 f'{time_pass}\n')
        f2.close()
        print('\rStart saving checkpoint [DONE]')

    writer.close()


def get_epoch_model():
    if os.path.exists(last_model):
        with open(epoch_file) as f:
            epoch = int(f.readline())
        return epoch + 1, last_model
    else:
        return 0, None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if False and torch.cuda.is_available() and False else 'cpu')
    logging.info(f'Using device {device}')

    start_epoch, model_path = get_epoch_model()
    net = UNet(n_channels=n_channels, n_classes=n_classes)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if model_path is not None:
        net.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f'Model loaded from {model_path}')

    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    net.to(device=device)

    try:
        train_net(net=net,
                  start_epoch=start_epoch,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            sys.exit(0)
