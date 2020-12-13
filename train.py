import numpy as np
import torch
from model import UNet
from eda import prep_dataset, get_batch
from torch.utils.data import DataLoader
from tversky_loss import TverskyLoss
import wandb
import time


device = torch.device('cuda' if False else 'cpu')
print(f"Using {device}")

dataset = np.load('data_pub.zip')


model = UNet(in_dim=4, out_dim=1, num_filters=1)
print(f'Initialized Model w/ : {sum(p.numel() for p in model.parameters() if p.requires_grad)} params')
model.to(device)

epochs = 100
batch_size = 1
alpha= .75
beta = .75
lr = .05

use_wandb = False
if use_wandb:
    wandb.init(project="cs446", entity="weustis")
    wandb.watch(model)


crit = TverskyLoss(alpha, beta)
opt = torch.optim.Adam(model.parameters(), lr=lr)


for epoch in range(epochs):
    t0 = time.time()
    opt.zero_grad()

    X, y = get_batch(dataset, batch_size, device)
    t1 = time.time()
    y_pred = model(X)
    t2 = time.time()
    loss = crit(y_pred, y)

    if use_wandb:
        wandb.log({"Train": loss})

    loss.backward()
    t3 = time.time()
    opt.step()

    print(f'{format(Loss), ".3f"} | Total: {format(t3-t0), ".3f"} | Data: {format(t1-t0), ".3f"} | Pred: {format(t2-t1), ".3f"}| Back: {format(t3-t2), ".3f"}')