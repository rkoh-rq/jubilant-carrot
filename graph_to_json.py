import argparse
import pandas as pd
import gzip
import array
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
from scipy.sparse import coo, coo_matrix, vstack
from sklearn.decomposition import TruncatedSVD

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    try:
      a.fromfile(f, 4096)
    except:
      break
    yield asin, a.tolist()

def related_convert_to_list(x):
  r = set([])
  if x.isna()['related']:
    return r
  else:
    x = x['related']
    for key in x:
      for item in x[key]:
        r.add(item)
    return r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='Musical_Instruments')
    parser.add_argument('--reduced_dim', type=int, default=128)
    args = parser.parse_args()

    ratings_df = pd.read_csv('ratings_{}.csv'.format(args.category), header=None)
    ratings_df = ratings_df[ratings_df.groupby(0)[1].transform('count')>=5]
    ratings_df = ratings_df.reset_index(drop=True)

    movies_ratings_df = pd.read_csv('ratings_Movies_and_TV.csv', header = None)

    image_features_index = {feature[0]:i for i,feature in enumerate(readImageFeatures('image_features_{}.b'.format(args.category)))}
    image_features = []
    for feature in readImageFeatures('image_features_{}.b'.format(args.category)):
        image_features.append(coo_matrix(feature[1]))
        
    image_feat_mx = vstack(image_features)
    clf = TruncatedSVD(args.reduced_dim)
    Xpca = clf.fit_transform(image_feat_mx)
    
    meta_df = getDF('meta_{}.json.gz'.format(args.category))
    meta_df = meta_df[['asin', 'related']]
    meta_df = meta_df.set_index('asin')
    meta_df = meta_df.apply(lambda x: related_convert_to_list(x), axis=1)
    G = nx.Graph()

    reviewers = list(set(ratings_df[0]))
    set_products = set(ratings_df[1])
    products = list(set_products)

    val = int(len(reviewers) * 0.7)
    test = int(len(reviewers) * 0.85)

    
    id_map = {}
    for i, reviewer in enumerate(reviewers):
        id_map[reviewer] = i
    p = len(reviewers)

    for i, product in enumerate(products):
        id_map[product] = i + p

    with open('{}-id_map.json'.format(args.category), 'w') as outfile:
        json.dump(id_map, outfile)


    image_feat_np = np.zeros((len(products) + len(reviewers), args.reduced_dim))
    for i, product in enumerate(products):
      if product in image_features_index:
        image_feat_np[i + p, :] = Xpca[image_features_index[product]]

    with open('{}-feats.npy'.format(args.category), 'wb') as f:
        np.save(f, image_feat_np)
                             
    # Train
    for reviewer in reviewers[:val]:
        G.add_node(reviewer, test=False, val=False, label=[1,0], feature=[0 for i in range(32)])
    # Validation
    for reviewer in reviewers[val:test]:
        G.add_node(reviewer, test=False, val=True, label=[1,0], feature=[0 for i in range(32)])
    # Test
    for reviewer in reviewers[test:]:
        G.add_node(reviewer, test=True, val=False, label=[1,0], feature=[0 for i in range(32)])

    for product in products:
        if product in image_features_index:
            G.add_node(product, test=False, val=False, label=[0,1], feature=Xpca[image_features_index[product]])
        else:
            G.add_node(product, test=False, val=False, label=[0,1], feature=[0 for i in range(32)])

    for row in ratings_df.iterrows():
        G.add_edge(row[1][0], row[1][1])

    for related in meta_df.index:
        if related in set_products:
            for r in meta_df[related]:
                if r in set_products:
                    G.add_edge(related, r)

    data = json_graph.node_link_data(G)
    with open('{}-G.json'.format(args.category), 'w') as outfile:
        json.dump(data, outfile)
    
    class_map = {}

    for reviewer in reviewers:
        class_map[reviewer] = [0, 1, 0]

    for reviewer in set(ratings_df[0]).intersection(movies_ratings_df[0]):
                class_map[reviewer] = [1, 0, 0]

    for product in products:
        class_map[product] = [0, 0, 1]
    with open('{}-class_map.json'.format(args.category), 'w') as outfile:
        json.dump(class_map, outfile)
