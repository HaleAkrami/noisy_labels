##This code is from : https://github.com/HanxunH/Active-Passive-Losses
import torch
import torch.nn.functional as F
import numpy as np

class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, device, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, device, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes,device, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae.mean()

class BetaCrossEnropyError(torch.nn.Module):
    def __init__(self, num_classes,device, scale=1.0,beta=0.1):
        super(BetaCrossEnropyError, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        self.beta=beta
        return

    def forward(self, pred, labels):
        #max_val = pred.max(dim=1, keepdim=True)[0]
        #max_val2 = torch.argmax(x[:, st:ed], dim=-1)
        #probs = F.softmax(pred - max_val, dim=1).to(self.device)
        #eps = 1e-8
        #label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)

        #single_prob = torch.sum(probs * label_one_hot, dim=1)
        #single_prob = single_prob + (single_prob < eps) * eps
        #part1 = (self.beta + 1) /((self.beta) * (single_prob ** self.beta - 1)+eps)

        #part2 = (probs ** (self.beta + 1)).sum(dim=1, keepdims=True)
        #bce=torch.mean(- part1 + part2)


        probs = F.softmax(pred,dim=1).to(self.device)

        ns = labels.shape[0]

        C = -(self.beta + 1) / self.beta


        term1 = C * (torch.pow(probs[range(ns), labels], self.beta) - 1 )  # This needs to be checked!!!!!!!!!
        term2 = torch.sum(torch.pow(probs , self.beta + 1), dim=1)

        bce = torch.mean(term1 + term2)
        #
        return self.scale * bce


