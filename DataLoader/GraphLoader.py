from collections import deque
from numpy import zeros, ones, delete
from numpy.random import shuffle
import networkx as nx
import pandas as pd
from .data_tools import getDF, readImageFeatures, add_to_queue_or_graph, related_convert_to_list
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo, coo_matrix, vstack
import torch


class GraphLoader():
    def __init__(self, category, batch_size=64, min_reviews=5, reduced_dim=64):
        self.reduced_dim = reduced_dim
        self.batch_size = batch_size

        ratings_df = pd.read_csv('ratings_{}.csv'.format(category), header=None)
        ratings_df = ratings_df[ratings_df.groupby(0)[1].transform('count')>=min_reviews]
        ratings_df = ratings_df.reset_index(drop=True)

        self.ratings_df_reviewer = ratings_df.set_index(0)
        self.ratings_df_asin = ratings_df.set_index(1)
        self.ratings_df_index = {id:i for i,id in enumerate(self.ratings_df_reviewer.index)}
        self.ratings_df_index_i = self.ratings_df_reviewer.index

        movies_ratings_df = pd.read_csv('ratings_Movies_and_TV.csv', header = None)

        # Generate labels
        num_positive = 0
        self.labels = zeros(len(self.ratings_df_index_i))
        is_positive = self.ratings_df_index_i.intersection(movies_ratings_df[0])
        for user in is_positive:
            self.labels[self.ratings_df_index[user]] = 1
            num_positive += 1
            
        num_negative = len(self.labels) - num_positive
        # Drop some negative samples
        negative_to_drop = [x for x,y in enumerate(self.ratings_df_index_i) if self.labels[self.ratings_df_index[y]] == 0]
        shuffle(negative_to_drop)
        negative_to_drop = negative_to_drop[:max(0, num_negative - num_positive)]
        print(len(negative_to_drop))
        idx_list = [i for i in range(len(self.labels))]
        idx_list = delete(idx_list, negative_to_drop)

        meta_df = getDF('meta_{}.json.gz'.format(category))
        meta_df = meta_df[['asin', 'related']]
        meta_df = meta_df.set_index('asin')
        meta_df = meta_df.apply(lambda x: related_convert_to_list(x), axis=1)
        self.meta_df = meta_df
        
        self.image_features_index = {feature[0]:i for i,feature in enumerate(readImageFeatures('image_features_{}.b'.format(category)))}
        self.image_features = []
        for feature in readImageFeatures('image_features_{}.b'.format(category)):
            self.image_features.append(coo_matrix(feature[1]))

        image_feat_mx = vstack(self.image_features)
        clf = TruncatedSVD(self.reduced_dim)
        self.image_features = clf.fit_transform(image_feat_mx)

            
        self.idx_train, self.idx_val, self.idx_test = self.train_val_test_split(idx_list)
            
        self.mode = 'train'
        self.idx = self.idx_train

    def train_val_test_split(self, idx_list, split_train=0.6, split_val=0.2):
        shuffle(idx_list)
        num_total = len(idx_list)

        num_train = int(split_train * num_total)
        num_val = int((split_train + split_val) * num_total)

        idx_train = idx_list[:num_train]
        idx_val = idx_list[num_train:num_val]
        idx_test = idx_list[num_val:]

        return idx_train, idx_val, idx_test


    def set_mode(self, mode):
        self.mode = mode
        if self.mode == 'train':
            shuffle(self.idx_train)
            self.idx = self.idx_train
        elif self.mode == 'val':
            self.idx = self.idx_val
        else:
            self.idx = self.idx_test

    def __iter__(self):
        i = 0
        for i in range(len(self.idx)):
            yield self.get_graph_around_user(self.idx[i])

    def __len__(self):
        # Number of users
        return len(self.idx)

    def get_graph_around_user(self, user_idx):
        '''
        Return a adjacency matrix and feature matrix of graph surrounding the user
        '''
        user = self.ratings_df_index_i[user_idx]
        features = zeros((50, self.reduced_dim))
        G = nx.Graph()
        Q = deque()
        Q.append((user, 0)) # 0 for user
        idx = 0

        while Q and G.number_of_nodes() < 50:
            node_name, node_type = Q.popleft()
            if node_name not in G._node:
                if node_type == 1 and bytes(node_name, 'utf-8') in self.image_features_index.keys():
                    features[G.number_of_nodes(), :] = self.image_features[self.image_features_index[bytes(node_name, 'utf-8')]]
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

        adj = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
        adj_pad = zeros((50, 50))
        adj_pad[:adj.shape[0], :adj.shape[1]] = adj
        return torch.Tensor(adj_pad), torch.Tensor(features), torch.LongTensor([self.labels[user_idx]]), G.number_of_nodes()
