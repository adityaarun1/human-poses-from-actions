from __future__ import print_function
import torchvision.models as models
import ref
import torch
import torch.nn as nn
import os
import torchvision.models as models
from models.hg import HourglassNet, PredictionNet, ConditionalNet


#Re-init optimizer
def getModel(opt):
    if 'pred_net' in opt.arch:
        model = PredictionNet(opt.nStack, opt.nModules, opt.nFeats,
                              opt.numOutput)
    elif 'cond_net' in opt.arch:
        model = ConditionalNet(opt.nStack, opt.nModules, opt.nFeats,
                               opt.numOutput, opt.nActionClass)
    elif 'hg' in opt.arch:
        model = HourglassNet(opt.nStack, opt.nModules, opt.nFeats,
                             opt.numOutput)
    else:
        raise Exception('Model name not known')
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        opt.LR,
        alpha=ref.alpha,
        eps=ref.epsilon,
        weight_decay=ref.weightDecay,
        momentum=ref.momentum)

    if opt.loadModel != 'none':
        checkpoint = torch.load(opt.loadModel)
        if type(checkpoint) == type({}):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint.state_dict()
        model.load_state_dict(state_dict)

    return model, optimizer


def saveModel(model, optimizer, path):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)
