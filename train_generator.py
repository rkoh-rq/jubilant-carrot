from Model.Generator import Generator
import torch.optim as optim
import torch
from tqdm import tqdm
import copy
from DataLoader.GraphLoader import GraphLoader
import argparse


def train_generator(epoch, g, max_gen_step):
    converted = 0
    negative = 0
    for i, data in enumerate(tqdm(graph_loader, total=len(graph_loader))):
        adj, feat, label, num_nodes = data
        if args.cuda:
            adj = adj.cuda()
            feat = feat.cuda()
            label = label.cuda()
        f = g.model(feat, adj)
        if torch.argmax(f) == 0:
            g.set_graph(adj, feat, num_nodes)
            negative += 1
            for i in range(max_gen_step):

                optimizer.zero_grad()
                G = copy.deepcopy(g.G)
                p_start, a_start, p_end, a_end, G = g.forward(G)

                Rt, isClass = g.calculate_reward(G)
                loss = g.calculate_loss(Rt, p_start, a_start, p_end, a_end, G)
                loss.backward()
                optimizer.step()

                if isClass:
                    converted += 1
                    break

    print('Epoch {}:\t Converted {} out of {} initially negative nodes'.format(epoch, converted, negative))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for training')
    parser.add_argument('--modelpath', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.99)
    parser.add_argument('--hyp1', type=float, default=1)
    parser.add_argument('--hyp2', type=float, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--rollout', type=int, default=10)
    parser.add_argument('--max_gen_step', type=int, default=30)
    parser.add_argument('--reduced_dim', type=int, default=128, help='Reduces the number of features via PCA to this number before passing as input')
    parser.add_argument('--cuda', action='store_true',
                        help='Enable CUDA training (only if available, otherwise behaves as False')
    parser.add_argument('--category', type=str, default='Musical_Instruments')
    args = parser.parse_args()
    
    MAXNUMNODES = 50
    g = Generator(args.modelpath, MAXNUMNODES, cuda=args.cuda, c=1, hyp1=args.hyp1, hyp2=args.hyp2, nfeat=args.reduced_dim)
    graph_loader = GraphLoader(args.category, reduced_dim=args.reduced_dim)
    optimizer = optim.Adam(g.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    for e in range(args.epochs):
        train_generator(e, g, args.max_gen_step)
