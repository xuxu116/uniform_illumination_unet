import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import numpy as np

from unet import UNet
from data_loading import BasicDataset, BasicTestDataset


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        postfix = ""
):
    dataset = BasicDataset()
    train_loader = DataLoader(dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              num_workers=16, 
                              pin_memory=True)
    
    testdataset = BasicTestDataset()
    test_loader = DataLoader(testdataset, 
                              shuffle=False, 
                              batch_size=4, 
                              num_workers=8, 
                              pin_memory=True)
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Define scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=0.00001)
    global_step = 0
    lossl1 = nn.L1Loss(reduction="none")
    mseloss = torch.nn.MSELoss(reduction="none")
    weight = 100 * torch.tensor([1,0.5,0.2]).reshape([1,3,1,1]).to(device)

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                source_lab, mask, offset, gt = batch
                source_lab = source_lab.to(device, memory_format=torch.channels_last)
                mask = mask.to(device, memory_format=torch.channels_last)
                offset = offset.to(device, memory_format=torch.channels_last)

                output = model(source_lab)
                loss_l1 = mask*(lossl1(output, offset)) * weight
                loss_l1 = loss_l1.mean()
                loss_l1norm = torch.abs(mask*output[:,1:3,:,:]).mean()
                loss = loss_l1 #+ loss_l1norm * 0.01

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                scheduler.step()

                pbar.update(source_lab.shape[0])
                pbar.set_postfix(**{'lossl1': loss_l1.item(), 'lossl1norm': loss_l1norm.item()})
                # break
        """
        source_lab_np, mask_np, output_np, gt_np, offset_np = source_lab.detach().cpu().numpy(), mask.detach().cpu().numpy(), output.detach().cpu().numpy(), gt.detach().cpu().numpy(), offset.detach().cpu().numpy()
        source_lab_np = source_lab_np.transpose(0, 2, 3, 1)
        output_np = output_np.transpose(0, 2, 3, 1)
        gt_np = gt_np.transpose(0, 2, 3, 1)
        offset_np = offset_np.transpose(0, 2, 3, 1)

        target = source_lab_np + output_np
        align = source_lab_np + offset_np

        for i in range(len(target)):
            source_cv = cv2.cvtColor(np.clip(source_lab_np[i]*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)
            target_cv = cv2.cvtColor(np.clip(target[i]*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)
            gt_cv = cv2.cvtColor(np.clip(gt_np[i]*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)

            align_cv = cv2.cvtColor(np.clip(align[i]*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)

            vis = np.concatenate([source_cv, target_cv, align_cv, gt_cv], axis=1)

            cv2.imwrite("./vis/epoch_%d_%d.jpg"%(epoch, i), vis)
        """
        model.eval()
        test_path = "./test_result/%s/epoch_%s"%(postfix, epoch)
        os.makedirs(test_path, exist_ok=True)
        for batch in tqdm(test_loader):
            source_lab, names, sources, masks = batch
            # if "911_1" not in names:
            #     continue

            source_lab = source_lab.to(device, memory_format=torch.channels_last)
            output = model(source_lab)
            source_lab_np, output_np, sources, masks = source_lab.detach().cpu().numpy(), output.detach().cpu().numpy(),  sources.detach().cpu().numpy(),  masks.detach().cpu().numpy()
            source_lab_np = source_lab_np.transpose(0, 2, 3, 1)
            output_np = output_np.transpose(0, 2, 3, 1)
            # target = source_lab_np + output_np

            for i in range(len(sources)):
                source = sources[i]
                mask = masks[i]
                mask_uint8 = np.clip(mask*255, 0 ,255).astype('uint8')

                source_lab_mask = source.astype("float32") / 255 * mask
                source_lab_mask = np.clip(source_lab_mask*255, 0 ,255).astype('uint8')
                source_lab_mask = cv2.cvtColor(source_lab_mask, cv2.COLOR_BGR2LAB).astype("float32") / 255

                source_lab_np2 = source_lab_mask + output_np[i]
                source_lab_np2 = cv2.cvtColor(np.clip(source_lab_np2*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)

                (x, y, w, h) = cv2.boundingRect(mask_uint8)
                center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
                result1 = cv2.seamlessClone(source_lab_np2, source, mask_uint8, center_face2, cv2.NORMAL_CLONE)

                source_cv = cv2.cvtColor(np.clip(source_lab_np[i]*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)
                # target_cv = cv2.cvtColor(np.clip(target[i]*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)

                # vis = np.concatenate([source, result1, source_lab_np2, np.repeat(mask_uint8, 3, axis=2)], axis=1)
                vis = np.concatenate([source,  result1], axis=1)
                cv2.imwrite(os.path.join(test_path, "%s.jpg"%(names[i])), vis)

        # state_dict = model.state_dict()
        torch.save(state_dict, str('./ckpts/{}_checkpoint_epoch{}.pth'.format(postfix, epoch)))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--postfix', '-p', type=str, default="", help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # Apply Kaiming initialization to all layers
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0)
                
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp,
        postfix=args.postfix
    )