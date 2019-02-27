import numpy as np
import tensorflow as tf
import os
import time
from glob import glob
from PIL import Image
import csv


PATH = '/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/Gender_CNN/data/resized/'

data_types = ['train', 'test']

for d in data_types:

    print('[{}] collecting data...'.format(d))
    filenames = []
    count = 0
    for root, dirs, files in os.walk(PATH):
        count += 1
        filenames.extend([f.replace(PATH, '') for f in glob(os.path.join(root, d, '*.jpg'))])
        print('================['+ str(count) +']================')

    filenames.sort()

    print('[{}] collecting labels...'.format(d))
    labels_dico = {}
    with open(os.path.join(PATH, 'identity_meta.csv'), 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            seq, _, _, _, gender = row
            labels_dico[seq] = 0 if gender.strip() == 'm' else 1

    
    labels = [labels_dico[f.split('/')[0]] for f in filenames]

    print('[{}] saving...'.format(d))
    # Save
    with tf.python_io.TFRecordWriter(os.path.join(PATH, 'vggface2_dataset_{}.tfrecords'.format(d))) as writer:
        for index, filename in enumerate(filenames):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
                        'gender': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]]))
                    }))
            writer.write(example.SerializeToString())


print('done!')
