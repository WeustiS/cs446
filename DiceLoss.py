import torch.nn as nn
import torch.nn.functional as F
import torch
class DiceBCELoss(nn.Module):
    def __init__(self, alpha, beta, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):


        # flatten label and prediction tensors

        targets = torch.eye(4)[targets.type(torch.LongTensor)]
        targets = targets.permute(0, -1, 2, 3, 4, 1).squeeze(-1).type(torch.FloatTensor).cuda()
        inputs = inputs.cuda()
        

        mask = torch.zeros(1, 4, 192, 192, 192).cuda()
        mask[:, 1:, 20:150, :170, :150] = 1
        inputs = inputs * mask
        targets = targets * mask

        #inputs = inputs[:, 1:, :, :, :]
        #targets = targets[:, 1:, :, :, :]
        return self.tversky(inputs, targets)

    def tversky(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        print(TP.item(), FP.item(), FN.item())
        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return 1 - Tversky

