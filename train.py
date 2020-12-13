import numpy as np
import torch
from model import UNet
from eda import prep_dataset, get_batch
from torch.utils.data import DataLoader
from tversky_loss import TverskyLoss
import wandb
import time
import tqdm


device = torch.device('cuda' if True else 'cpu')
print(f"Using {device}")

dataset = np.load('data_pub.zip')


model = UNet(in_dim=4, out_dim=1, num_filters=1)
print(f'Initialized Model w/ : {sum(p.numel() for p in model.parameters() if p.requires_grad)} params')
model.to(device)

epochs = 5000
batch_size = 64
alpha= 1
beta = .5
lr = .05
use_wandb = True


if use_wandb:
    wandb.init(project="cs446", entity="weustis")
    wandb.watch(model)


crit = TverskyLoss(alpha, beta)
opt = torch.optim.Adam(model.parameters(), lr=lr)

data = [np.load('data_pub/train/001_imgs.npy'), np.load('data_pub/train/001_seg.npy')]
for epoch in tqdm.tqdm(range(epochs)):

    opt.zero_grad()
    try:
        X, y = get_batch(dataset, batch_size, device)
    except:
        print("Failed getting dataset, trying again")
        X, y = get_batch(dataset, batch_size, device)

    y_pred = model(X)

    loss = crit(y_pred, y)

    if use_wandb:
        wandb.log({"Train": loss})

    loss.backward()
    opt.step()

    print(f'        {format(loss, ".3f")}')

torch.save(model, "./UNet_Base")