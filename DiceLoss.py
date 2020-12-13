import torch.nn as nn
import torch.nn.functional as F
import torch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors

        inputs = inputs.view(-1).cuda()
        targets = targets.view(-1).type(torch.LongTensor)
        targets = F.one_hot(targets, num_classes=4)
        targets = targets.view(-1).type(torch.FloatTensor).cuda()

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection) / (inputs.sum() + targets.sum())
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE