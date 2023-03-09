import os
import sys
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import dataloader
import data_generator
from models import resnet, NCM
from metric import AverageMeter, Logger
from utils import get_feature_size

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--device_name', type=str, default='hspark')
# Dataset Settings
parser.add_argument('--root', type=str, default='./data/')
parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'HAR'])
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=0)
# Model Settings
parser.add_argument('--model_name', type=str, default='ResNet18', choices=['ResNet18', 'ImageNet_ResNet18'])
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--classifier', type=str, default='FC', choices=['FC', 'NCM'])
# CL Settings
parser.add_argument('--class_increment', type=int, default=1)

args = parser.parse_args()


def train(epoch, model, train_loader, criterion, optimizer, classifier=None):
    model.train()

    acc = AverageMeter()
    losses = AverageMeter()

    num_iter = math.ceil(len(train_loader.dataset) / args.batch_size)
    l2_lambda = 2e-4

    for batch_idx, (x, y) in enumerate(train_loader):
        y = y.type(torch.LongTensor)
        x, y = x.to(args.device).float(), y.to(args.device)

        if args.classifier == 'NCM':
            features = model.features(x)
            classifier.train_(features, y)
            logits = classifier.evaluate_(features)
        else:
            logits = model(x) # FC

        loss = criterion(logits, y)

        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + (l2_norm * l2_lambda)

        _, predicted = torch.max(logits, dim=1)

        # Compute Gradient and do SGD step
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        losses.update(loss)

        correct = predicted.eq(y).cpu().sum().item()
        acc.update(correct, len(y))

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.2f Accuracy: %.2f' % (args.dataset, epoch+1, args.epoch, batch_idx+1, num_iter, loss, acc.avg*100))
        sys.stdout.flush()

    return loss.item(), acc.avg*100

def test(task, model, test_loader, classifier=None):
    acc = AverageMeter()
    sys.stdout.write('\n')

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device).float(), y.to(args.device)

            if args.classifier == 'NCM':
                features = model.features(x)
                logits = classifier.evaluate_(features)
            else:
                logits = model(x) # FC

            _, predicted = torch.max(logits, dim=1)

            correct = predicted.eq(y).cpu().sum().item()
            acc.update(correct, len(y))

            sys.stdout.write('\r')
            sys.stdout.write("Test | Accuracy (Test Dataset Up to Task-%d): %.2f%%" % (task+1, acc.avg*100))
            sys.stdout.flush()

    return acc.avg*100

def main():
    ## GPU Setup
    device = 'cuda:' + args.device
    args.device = torch.device(device)
    torch.cuda.set_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Dataset Generator
    if 'CIFAR' in args.dataset:
        data_generator.__dict__['Generator'](args)
        if args.dataset == 'CIFAR10': args.num_classes = 10
        else: args.num_classes = 100

    # Create Model
    model_name = args.model_name
    if 'ResNet' in model_name:
        model = resnet.__dict__[args.model_name](args.num_classes)
        model.to(args.device)

    # Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    feature_size = get_feature_size(model_name)

    # Select Classifier
    classifier_name = args.classifier
    if classifier_name == 'NCM':
        classifier = NCM.NearestClassMean(feature_size, args.num_classes, device=args.device)
    else:
        classifier = None # FC layer fine-tuning

    # For plotting the logs
    logger = Logger('logs/' + args.dataset + '/' + args.device_name, args.classifier)
    log_t = 1
    
    data_loader = dataloader.dataloader(args)
    last_test_acc = 0

    for idx in range(0, args.num_classes, args.class_increment):
        task = [k for k in range(idx, idx+args.class_increment)]
        print('\nTask : ', task)

        train_loader = data_loader.load(task)
        test_loader = data_loader.load(task, train=False)

        best_acc = 0
        for epoch in range(args.epoch):
            loss, train_acc = train(epoch, model, train_loader, criterion, optimizer, classifier)

            if train_acc > best_acc:
                best_acc = train_acc
                logger.result('Train Epoch Loss/Labeled', loss, epoch)

            if classifier_name == 'NCM' and epoch+1 != args.epoch:
                classifier.update_mean(task)

        logger.result('Train Accuracy', best_acc, log_t)

        test_acc = test(idx, model, test_loader, classifier)
        logger.result('Test Accuracy', test_acc, log_t)
        last_test_acc = test_acc

        log_t += 1

    logger.result('Final Test Accuracy', last_test_acc, 1)
    # the average test accuracy over all tasks
    print("\n\nAverage Test Accuracy : %.2f%%" % last_test_acc)

    args.device = device
    metric_dict = {'metric': last_test_acc}
    logger.config(config=args, metric_dict=metric_dict)


if __name__ == '__main__':
    main()