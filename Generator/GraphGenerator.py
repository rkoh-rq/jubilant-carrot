from collections import deque
from tools import train_val_test_split
from numpy import zeros, ones
from numpy.random import shuffle
import networkx as nx
import pandas as pd
from .data_tools import getDF, readImageFeatures, add_to_queue_or_graph, related_convert_to_list
from scipy.sparse import coo, coo_matrix
import torch

class GraphGenerator():
    def __init__(self, category, batch_size=20, min_reviews=3):
        self.batch_size = batch_size

        ratings_df = pd.read_csv('ratings_{}.csv'.format(category), header=None)
        ratings_df = ratings_df[ratings_df.groupby(0)[1].transform('count')>=min_reviews]
        ratings_df = ratings_df.reset_index(drop=True)

        self.ratings_df_reviewer = ratings_df.set_index(0)
        self.ratings_df_asin = ratings_df.set_index(1)
        self.ratings_df_index = {id:i for i,id in enumerate(ratings_df[0])}
        self.ratings_df_index_i = ratings_df[0]

        movies_ratings_df = pd.read_csv('ratings_Movies_and_TV.csv', header = None)

        # Generate labels
        self.labels = zeros((len(self.ratings_df_index_i), 2))
        self.labels[:, 0] = ones(len(self.ratings_df_index_i))
        for user in set(ratings_df[0]).intersection(movies_ratings_df[0]):
            self.labels[self.ratings_df_index[user]][1] = 1
            self.labels[self.ratings_df_index[user]][0] = 0

        meta_df = getDF('meta_{}.json.gz'.format(category))
        meta_df = meta_df[['asin', 'related']]
        meta_df = meta_df.set_index('asin')
        meta_df = meta_df.apply(lambda x: related_convert_to_list(x), axis=1)
        self.meta_df = meta_df
        
        self.image_features_index = {feature[0]:i for i,feature in enumerate(readImageFeatures('image_features_{}.b'.format(category)))}
        self.image_features = []
        for feature in readImageFeatures('image_features_{}.b'.format(category)):
            self.image_features.append(coo_matrix(feature[1]))
            
        self.idx_train, self.idx_val, self.idx_test = self.train_val_test_split(len(self.ratings_df_index.keys()))
            
        self.mode = 'train'

    def train_val_test_split(self, num_total, split_train=0.6, split_val=0.2):
        idx_list = [i for i in range(num_total)]
        shuffle(idx_list)

        num_train = int(split_train * num_total)
        num_val = int((split_train + split_val) * num_total)

        idx_train = idx_list[:num_train]
        idx_val = idx_list[num_train:num_val]
        idx_test = idx_list[num_val]

        return idx_train, idx_val, idx_test

    def __iter__(self):
        if self.mode == 'train':
            shuffle(self.idx_train)
            idx = self.idx_train
        elif self.mode == 'val':
            idx = self.idx_val
        else:
            idx = self.idx_test
        i = 0
        for i in range(len(idx)):
            if i % self.batch_size == 0:
                adj_, feat_, label_ = self.get_graph_around_user(idx[i])
                adj = adj_.unsqueeze(0)
                feat = feat_.unsqueeze(0)
                label = label_.unsqueeze(0)
            else:
                adj_, feat_, label_ = self.get_graph_around_user(idx[i])
                adj = torch.vstack((adj, adj_))
                feat = torch.vstack((feat, feat_))
                label = torch.vstack((label, label_))
            
            if i % self.batch_size == self.batch_size - 1:
                yield adj, feat, label

    def __len__(self):
        # Number of users
        return len(self.ratings_df_index.keys())

    def get_graph_around_user(self, user_idx):
        '''
        Return a adjacency matrix and feature matrix of graph surrounding the user
        '''
        user = self.ratings_df_index_i[user_idx]
        features = zeros(50, 4096)
        G = nx.Graph()
        Q = deque()
        Q.append((user, 0)) # 0 for user
        idx = 0

        while Q and G.number_of_nodes() < 50:
            node_name, node_type = Q.popleft()
            if node_name not in G._node:
                if node_type == 1:
                    features[G.number_of_nodes(), :] = self.image_features[self.image_features_index[node_name]]
                G.add_node(node_name)

            if node_type == 0: # User
                products = self.ratings_df_reviewer.loc[node_name][1]
                add_to_queue_or_graph(Q, G, node_name, products, 1)

            else:
                if node_name in self.meta_df.index:
                    products = self.meta_df.loc[node_name]
                    add_to_queue_or_graph(Q, G, node_name, products, 1)

                if node_name in self.ratings_df_asin.index:
                    users = self.ratings_df_asin.loc[node_name][0]
                    add_to_queue_or_graph(Q, G, node_name, users, 0)

        return torch.Tensor(nx.linalg.graphmatrix.adjacency_matrix(G)), torch.Tensor(features), torch.Tensor(self.labels[user_idx])

    def add_to_queue_or_graph(self, Q, G, source, to_add, indicator):
        if isinstance(to_add, str):
            if to_add not in G._node:
                Q.append((to_add, indicator))
            else:
                G.add_edge(source, to_add)
        else:
            for t in to_add:
                add_to_queue_or_graph(Q, G, source, t, indicator)