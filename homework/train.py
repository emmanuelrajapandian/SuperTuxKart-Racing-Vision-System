from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path

    device = torch.device('mps')
    model = Planner().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    train_data = load_data(
        '/Users/emmanuel/Downloads/UT-Austin/Deep-Learning/homework5/drive_data/train',
        transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(),
                                            dense_transforms.ColorJitter(brightness=0.7, contrast=0.8,
                                                                         saturation=0.7, hue=0.2),
                                            dense_transforms.ToTensor()]), )

    valid_data = load_data(
        '/Users/emmanuel/Downloads/UT-Austin/Deep-Learning/homework5/drive_data/valid',
        transform=dense_transforms.Compose([dense_transforms.ToTensor()]), )

    epochs = 150
    global_step = 0
    for epoch in range(epochs):
        model.train()

        for imgs, aim_points in train_data:
            imgs, aim_points = imgs.to(device), aim_points.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, aim_points)

            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, imgs, aim_points, outputs, global_step)

            if train_logger is not None:
                train_logger.add_scalar('train_loss', loss, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        model.eval()

        for imgs, aim_points in valid_data:
            imgs, aim_points = imgs.to(device), aim_points.to(device)

            outputs = model(imgs)
            val_loss = criterion(outputs, aim_points)

            if valid_logger is not None:
                log(valid_logger, imgs, aim_points, outputs, global_step)

            if valid_logger is not None:
                valid_logger.add_scalar('valid_loss', val_loss, global_step)

    save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
