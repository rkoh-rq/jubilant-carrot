from pandas import DataFrame
from array import array
import gzip

def add_to_queue_or_graph(Q, G, source, to_add, indicator):
  if isinstance(to_add, str):
    if to_add not in G._node:
      Q.append((to_add, indicator))
    else:
      G.add_edge(source, to_add)
  else:
    for t in to_add:
      add_to_queue_or_graph(Q, G, source, t, indicator)

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
  return DataFrame.from_dict(df, orient='index')

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array('f')
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