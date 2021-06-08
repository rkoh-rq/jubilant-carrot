from numpy.random import shuffle

def train_val_test_split(num_total, split_train=0.6, split_val=0.2):
  idx_list = [i for i in range(num_total)]
  shuffle(idx_list)

  num_train = int(split_train * num_total)
  num_val = int((split_train + split_val) * num_total)

  idx_train = idx_list[:num_train]
  idx_val = idx_list[num_train:num_val]
  idx_test = idx_list[num_val]

  return idx_train, idx_val, idx_test