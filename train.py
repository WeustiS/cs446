import numpy as np
import torch
from model import UNet
from eda import prep_dataset, get_batch
from torch.utils.data import DataLoader
from tversky_loss import TverskyLoss
import wandb
import time
import tqdm
from torch.optim.lr_scheduler import LambdaLR
from DiceLoss import DiceBCELoss

device = torch.device('cuda' if True else 'cpu')
print(f"Using {device}")

dataset = np.load('data_pub.zip')


model = UNet(in_dim=4, out_dim=4, num_filters=1)
print(f'Initialized Model w/ : {sum(p.numel() for p in model.parameters() if p.requires_grad)} params')
model.to(device)

epochs = 20
batch_size = 2
alpha= .5
beta = .5
lr = .05
use_wandb = True


if use_wandb:
    wandb.init(project="cs446", entity="weustis")
    wandb.watch(model)


crit = DiceBCELoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)

lambdalr = lambda epoch: .05 - (.0499*(epoch/epochs))
scheduler = LambdaLR(opt, lr_lambda=[lambdalr])
# data = [np.load('data_pub/train/001_imgs.npy'), np.load('data_pub/train/001_seg.npy')]



for epoch in tqdm.tqdm(range(epochs)):
    X, y = get_batch(dataset, 1, 'cpu')
    eloss = 0
    for batch in range(0, len(X) - batch_size, batch_size):

        x_batch = X[batch:batch + batch_size].to(device)
        opt.zero_grad()

        y_pred = model(x_batch)
        y_batch = y[batch:batch + batch_size].to(device)

        loss = crit(y_pred, y_batch)

        if use_wandb:
            wandb.log({"Train": loss})

        loss.backward()
        opt.step()
        eloss += loss.item()
        print(f'        {format(loss, ".3f")}')
    shuffle_idx = torch.randperm(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    if use_wandb:
        wandb.log({"Epoch Train": eloss})
    scheduler.step()

torch.save(model, "./UNet_Base")