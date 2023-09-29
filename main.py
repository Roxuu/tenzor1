import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ds, info = tfds.load('wine_quality', split=['train', 'test'], with_info=True)
train_ds, test_ds = ds['train'], ds['test']
print(info.features)
for example in train_ds.take(1):
    print(example)


    def get_features_and_labels(example: object) -> object:
        return example['features'], example['quality']


    train_features, train_labels = get_features_and_labels(train_ds)
