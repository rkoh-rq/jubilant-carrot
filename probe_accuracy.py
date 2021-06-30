import torch
from tqdm import tqdm
import copy
from DataLoader.GraphLoader import GraphLoader
import argparse
from Model.GCN import GCN


def probe(model, cuda):
    for mode in ['train', 'val', 'test']:
        pp = 0
        pn = 0
        np = 0
        nn = 0
        graph_loader.set_mode(mode)
        for i, data in enumerate(tqdm(graph_loader, total=len(graph_loader))):
            adj, feat, label, num_nodes = data
            if cuda:
                adj = adj.cuda()
                feat = feat.cuda()
            predicted = torch.argmax(model(feat, adj))
            if label == 0 and predicted == 0:
                nn += 1
            elif label == 0 and predicted == 1:
                np += 1
            elif label == 1 and predicted == 1:
                pp += 1
            elif label == 1 and predicted == 0:
                pn += 1

        print('Mode: {} \t PP:{} \t PN:{} \t NP:{} \t NN:{}'.format(mode, pp, pn, np, nn))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for training')
    parser.add_argument('--modelpath', type=str)
    parser.add_argument('--reduced_dim', type=int, default=128, help='Reduces the number of features via PCA to this number before passing as input')
    parser.add_argument('--category', type=str, default='Musical_Instruments')
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()
    
    model = GCN(nfeat=args.reduced_dim,
                nclass=2,
                dropout=0.1)
    model.load_state_dict(torch.load(args.modelpath))
    torch.no_grad()

    if args.cuda:
        model.cuda()
    model.eval()
    
    graph_loader = GraphLoader(args.category, reduced_dim=args.reduced_dim)
    probe(model, True if args.cuda else False)

