import numpy as np
import torch
from model import UNet
from eda import prep_dataset, get_batch, get_test, augmentClass
from torch.utils.data import DataLoader
from tversky_loss import TverskyLoss
import wandb
import time
import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from DiceLossTwo import DiceLoss
import random
import matplotlib.pyplot as plt 


device = torch.device('cuda' if True else 'cpu')
print(f"Using {device}")

dataset = np.load('data_pub.zip')


model = UNet(in_dim=4, out_dim=4, num_filters=5)
print(f'Initialized Model w/ : {sum(p.numel() for p in model.parameters() if p.requires_grad)} params')
model = model.to(device)

epochs = 100
batch_size = 1
alpha= .3
beta = .7
lr = .3
use_wandb = True


if use_wandb:
    wandb.init(project="cs446", entity="weustis")
    wandb.watch(model)


crit = DiceLoss(weight=np.array([.1, .3, .3, .3]))
opt = torch.optim.Adam(model.parameters(), lr=lr)


eloss = 0
min_test = .3
for epoch in tqdm.tqdm(range(epochs)):
    X, y = get_batch(dataset, 'cpu')

    X_t, y_t = get_test(dataset, 'cpu')
    eloss = 0
    opt.zero_grad()
    
    for batch in range(0, len(X) - batch_size, batch_size):
        x_batch = X[batch:batch + batch_size].to('cuda')
        y_batch = y[batch:batch + batch_size].to('cuda')

        y_pred = model(x_batch).to('cuda')

        
        loss, losses = crit(y_pred, y_batch)
        loss = loss.to(device)

        if use_wandb:
            wandb.log({"Train": loss})
            for i in range(len(losses)):
                wandb.log({f"Train Class {i}": losses[i].item()})

        loss.backward()

        if batch%(batch_size*64)==0:

            loss_t = 0
            losses = [0] * 4
            for batch in range(0, len(X_t), batch_size):
                pred_t = model(X_t[batch:batch+batch_size].cuda(),)
                loss_test, losses_test = crit(pred_t, y_t[batch:batch+batch_size])
                loss_t += loss_test.item()

                for i in range(len(losses_test)):
                    losses[i] += losses_test[i].item()/len(X_t)

            loss_t /= (len(X_t)/batch_size)
            if use_wandb:
                wandb.log({"Test": loss_t})
                for i in range(len(losses)):
                    wandb.log({f"Test Class {i}": losses[i]})

            print("TESTLOSS: " + str(loss_t) + "| Classes: " +  str([x.item() for x in losses_test]))

            if loss_t<min_test*.98:
                min_test = loss_t
                torch.save(model, f"./UNet_v6_Ckpt{min_test}")

           
        if batch%(16*batch_size) == 0:
            opt.step()
            opt.zero_grad()
        eloss += loss.item()
        print(f'        {format(loss, ".3f")}')

    shuffle_idx = torch.randperm(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    if use_wandb:
        wandb.log({"Epoch Train": eloss})

    
    torch.save(model, f"./UNet_v6_e{epoch}")

    for g in opt.param_groups:
        g['lr'] = .3 * .5**(epoch//33)