# jubilant-carrot
For URECA

Run the following (you will need around 1.4GB of space)

```
!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Musical_Instruments.csv
!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Musical_Instruments.json.gz
!wget http://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_Musical_Instruments.b
```

```
!wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv
```

Train the GNN
```
python train.py [-h] --experiment_name EXPERIMENT_NAME [--epochs EPOCHS] [--seed SEED] [--lr LR] [--batch_size BATCH_SIZE] [--dropout DROPOUT]
                [--weight_decay WEIGHT_DECAY] [--cuda] [--category CATEGORY] [--reduced_dim REDUCED_DIM] [--loss_weight LOSS_WEIGHT]
```

Train the generator
```
python train_generator.py [-h] --modelpath MODELPATH [--epochs EPOCHS] [--lr LR] [--b1 B1] [--b2 B2] [--hyp1 HYP1] [--hyp2 HYP2] [--dropout DROPOUT]
                          [--rollout ROLLOUT] [--max_gen_step MAX_GEN_STEP] [--reduced_dim REDUCED_DIM] [--cuda] [--category CATEGORY]
```

The notebook attached was done for initial data exploration and run on Google Colab
