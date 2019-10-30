from __future__ import division
from builtins import range
from past.utils import old_div
import torch
import numpy as np
from utils.utils import AverageMeter, Flip, ShuffleLR
from utils.eval import Accuracy, getPreds, finalPreds, maximumExpectedUtility
import cv2
import ref
from progress.bar import Bar
from utils.debugger import Debugger


def DiscoLoss(output, samples, targets, criterion):
    nb_feat_output = len(output)
    nb_samples = len(samples)
    nb_targets = len(targets)

    loss = criterion(output[0], targets[0])

    for i in range(nb_feat_output):
        for j in range(nb_targets):
            if i == 0 and j == 0:
                continue
            else:
                loss += criterion(output[i], targets[j])

    for i in range(nb_samples):
        for j in range(nb_targets):
            loss += criterion(samples[i], targets[j])
        for k in range(nb_samples):
            if i == k:
                continue
            else:
                loss += criterion(samples[i], samples[k])

    return loss


def step(split, epoch, opt, dataLoader, model, criterion, optimizer=None):
    if split == 'train':
        model.train()
    else:
        model.eval()
    Loss, Acc = AverageMeter(), AverageMeter()
    preds = []

    nIters = len(dataLoader)
    bar = Bar('{}'.format(opt.expID), max=nIters)

    for i, (input, targets, _, meta) in enumerate(dataLoader):
        input_var = torch.autograd.Variable(input).float().cuda(opt.GPU)
        target_var = []
        for t in range(len(targets)):
            target_var.append(
                torch.autograd.Variable(targets[t]).float().cuda(opt.GPU))
        z = []
        for k in range(opt.numNoise):
            noise = torch.autograd.Variable(
                torch.randn((input_var.shape[0], 1, 64, 64))).cuda(opt.GPU)
            z.append(noise)

        output, samples = model(input_var, z)
        pred_sample = maximumExpectedUtility(samples, criterion)
        target = maximumExpectedUtility(target_var, criterion)

        if opt.DEBUG >= 2:
            gt = getPreds(target.cpu().numpy()) * 4
            pred = getPreds((pred_sample.data).cpu().numpy()) * 4
            debugger = Debugger()
            img = (input[0].numpy().transpose(1, 2, 0) * 256).astype(
                np.uint8).copy()
            debugger.addImg(img)
            debugger.addPoint2D(pred[0], (255, 0, 0))
            debugger.addPoint2D(gt[0], (0, 0, 255))
            debugger.showAllImg(pause=True)

        loss = DiscoLoss(output, samples, target_var, criterion)

        Loss.update(loss.item(), input.size(0))
        Acc.update(
            Accuracy((pred_sample.data).cpu().numpy(),
                     (target.data).cpu().numpy()))
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            input_ = input.cpu().numpy()
            input_[0] = Flip(input_[0]).copy()
            inputFlip_var = torch.autograd.Variable(
                torch.from_numpy(input_).view(1, input_.shape[1], ref.inputRes,
                                              ref.inputRes)).float().cuda(
                                                  opt.GPU)
            _, samplesFlip = model(inputFlip_var, z)
            pred_sample_flip = maximumExpectedUtility(samplesFlip, criterion)
            outputFlip = ShuffleLR(
                Flip((pred_sample_flip.data).cpu().numpy()[0])).reshape(
                    1, ref.nJoints, ref.outputRes, ref.outputRes)
            output_ = old_div(((pred_sample.data).cpu().numpy() + outputFlip),
                              2)
            preds.append(
                finalPreds(output_, meta['center'], meta['scale'],
                           meta['rotate'])[0])

        Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f} ({Acc.val:.6f})'.format(
            epoch,
            i,
            nIters,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=Loss,
            Acc=Acc,
            split=split)
        bar.next()

    bar.finish()
    return {'Loss': Loss.avg, 'Acc': Acc.avg}, preds


def train(epoch, opt, train_loader, model, criterion, optimizer):
    return step('train', epoch, opt, train_loader, model, criterion, optimizer)


def val(epoch, opt, val_loader, model, criterion):
    return step('val', epoch, opt, val_loader, model, criterion)
