
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class Generator_loss(nn.Module):
    def __init__(self):
        super(Generator_loss, self).__init__()
        self.LAMBDA = 100

    def forward(self, disc_generated_output, gen_output, target):

        gan_loss = F.binary_cross_entropy_with_logits(torch.zeros_like(disc_generated_output), disc_generated_output, reduction='mean')
        l1_loss = torch.mean(torch.abs(target - gen_output))

        return gan_loss + (self.LAMBDA * l1_loss)

class Discriminator_loss(nn.Module):
    def __init__(self):
        super(Discriminator_loss, self).__init__()

    def forward(self, disc_real_output, disc_generated_output):

        real_loss = F.binary_cross_entropy_with_logits(torch.ones_like(disc_real_output), disc_real_output, reduction='mean')

        generated_loss = F.binary_cross_entropy_with_logits(torch.zeros_like(disc_generated_output), disc_generated_output, reduction='mean')

        return real_loss + generated_loss