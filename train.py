from __future__ import division
from __future__ import print_function

from DataLoader.GraphLoader import GraphLoader

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import argparse

from tqdm import tqdm

from Model.GCN import GCN
from tools import accuracy

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    graph_loader.set_mode('train')
    total_loss_train = 0
    output_list = []
    labels_list = []

    for i, data in enumerate(tqdm(graph_loader, total=len(graph_loader))):
        adj, feat, label, num_nodes = data
        if args.cuda:
            adj = adj.cuda()
            feat = feat.cuda()
            label = label.cuda()

        if i % args.batch_size == 0:
            current = model.forward(feat, adj)
            output = current.unsqueeze(0)
            labels = label
        else:
            current = model.forward(feat, adj)
            output = torch.vstack((output, current.unsqueeze(0)))
            labels = torch.vstack((labels, label))

        output_list.append(int(torch.argmax(current)))
        labels_list.append(int(label))

        if i % args.batch_size == args.batch_size - 1:
            labels = torch.flatten(labels)
            loss_train = F.cross_entropy(
                output, labels, weight=weights)
            total_loss_train += float(loss_train)
            # acc_train = accuracy(output, labels)
            loss_train.backward()
            optimizer.step()

    
    f1_train = f1_score(labels_list, output_list)
    acc_train = accuracy(output_list, labels_list)
    avg_loss_train = total_loss_train/len(graph_loader)
    
    total_loss_val = 0
    output_list = []
    labels_list = []
    model.eval()
    graph_loader.set_mode('val')
    
    for i, data in enumerate(tqdm(graph_loader, total=len(graph_loader))):
        adj, feat, label = data
        if args.cuda:
            adj = adj.cuda()
            feat = feat.cuda()
            label = label.cuda()

        if i % args.batch_size == 0:
            current = model.forward(feat, adj)
            output = current.unsqueeze(0)
            labels = label
        else:
            current = model.forward(feat, adj)
            output = torch.vstack((output, current.unsqueeze(0)))
            labels = torch.vstack((labels, label))
        if i % args.batch_size == args.batch_size - 1:
            labels = torch.flatten(labels)
            total_loss_val += float(F.cross_entropy(output, labels, weight=weights))
    
        output_list.append(int(torch.argmax(current)))
        labels_list.append(int(label))
        

    f1_val = f1_score(labels_list, output_list)
    acc_val = accuracy(output_list, labels_list)
    avg_loss_val = total_loss_val/len(graph_loader)
    
    
    print('\nEpoch: {:04d}'.format(epoch+1),
        # 'loss_train: {:.4f}'.format(loss_train.item()),
        'avg_loss_train: {:.4f}'.format(avg_loss_train),
        'f1_train: {:.4f}'.format(f1_train),
        'acc_train: {:.4f} ({} out of {} || {} out of {})'.format(acc_train[0], acc_train[1], acc_train[2], acc_train[3], acc_train[4]),
        # 'loss_val: {:.4f}'.format(loss_val.item()),
        'avg_loss_val: {:.4f}'.format(avg_loss_val),
        'f1_val: {:.4f}'.format(f1_val),
        'acc_val: {:.4f} ({} out of {} positive || {} out of {} negative)'.format(acc_val[0], acc_val[1], acc_val[2], acc_val[3], acc_val[4]),
        'time: {:.4f}s'.format(time.time() - t))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for training')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Used as file name extension for relevant files')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train for (default: 200)')
    parser.add_argument('--seed', type=int, default=200,
                        help='Seed for random functions (default: 200')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Enable CUDA training (only if available, otherwise behaves as False')
    parser.add_argument('--category', type=str, default='Musical_Instruments')
    parser.add_argument('--reduced_dim', type=int, default=128, help='Reduces the number of features via PCA to this number before passing as input')
    parser.add_argument('--loss_weight', type=int, default=0.1, help='Loss weight to address class imbalance. This is the weight of the first class')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    graph_loader = GraphLoader(args.category, reduced_dim=args.reduced_dim)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = GCN(nfeat=args.reduced_dim,
                nclass=2,
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)
    weights = torch.Tensor([args.loss_weight, 1-args.loss_weight])
    if args.cuda:
        model.cuda()
        weights = weights.to(device='cuda')

    for epoch in range(args.epochs):
        train(epoch)
        if epoch % 5 == 4:
            torch.save(model.state_dict(), '{}_ckpt_{}.pth'.format(args.experiment_name, epoch))
