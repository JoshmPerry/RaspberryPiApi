import numpy as np
import tensorflow as tf

def shuffle(data, labels):
    idx = np.random.permutation(len(data))
    x_all = np.array(data)[idx]
    y_all = np.array(labels)[idx]
    return x_all, y_all

def train_valid_split(raw_data, labels, split_ratio=0.8):
    split_index = int(split_ratio * len(raw_data))
    return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

def transform_to_dataset(data, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels)).batch(batch_size)
    return dataset

def arrange_data(data, labels):
    x_all, y_all = shuffle(data, labels)
    x_train, x_test, y_train, y_test = train_valid_split(x_all, y_all)
    train_dataset = transform_to_dataset(x_train, y_train)
    validation_dataset = transform_to_dataset(x_test, y_test)
    return train_dataset, validation_dataset