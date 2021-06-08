from __future__ import division
from __future__ import print_function

from Generator.GraphGenerator import GraphGenerator

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import argparse

from tqdm import tqdm

from Model import GCN, FocalLoss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for (default: 20)')
    parser.add_argument('--seed', type=int, default=200,
                        help='Seed for random functions (default: 200')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Enable CUDA training (only if available, otherwise behaves as False')
    parser.add_argument('--category', type=str, default='Musical_Instruments')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    graph_generator = GraphGenerator(args.category)
    idx_train, idx_val, idx_test = train_val_test_split(len(data_generator))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = GCN(nfeat=4096,
                nclass=2,
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()

    for epoch in range(args.epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        graph_generator.mode = 'train'
        total_loss_train = 0

        for feat, adj, labels in graph_generator:
            output = model(feat, adj)
            loss_train = F.cross_entropy(
                output, labels, weight=weights)
            total_loss_train += float(loss_train)
            # acc_train = accuracy(output, labels)
            loss_train.backward()
            optimizer.step()
        
            f1_train = f1_score(labels.cpu(), output)

        # model.eval()
        

        print('\nEpoch: {:04d}'.format(epoch+1),
            # 'loss_train: {:.4f}'.format(loss_train.item()),
            'avg_loss_train: {:.4f}'.format(total_loss_train/len(idx_train)),
            'f1_train: {:.4f}'.format(f1_train),
            # 'acc_train: {:.4f}'.format(acc_train.item()),
            # 'loss_val: {:.4f}'.format(loss_val.item()),
            # 'avg_loss_val: {:.4f}'.format(total_loss_val/len(idx_val)),
            # 'f1_val: {:.4f}'.format(f1_val),
            # 'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))

        # return loss_train, acc_train, loss_val, acc_val

