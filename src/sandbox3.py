import tensorflow as tf
import numpy as np
from dataset.DataSets2 import ds

def get_dict(database):
    xs,ys = database.NextTrainingBatch()
    return {x:xs, y_desired:ys}

train = ds.DataSets2('/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/Gender_CNN/data/...train')
test = ds.DataSets2('/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/Gender_CNN/data/...test')

with tf.nape_scope('input'):
    x = tf.placeholder(tf.float32, [None, None], name='x')
    y_desired = tf.placeholder(tf.float32, [None, 2], name='y_desired')

