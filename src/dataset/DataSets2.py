import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


class DataSets(object):
    def __init__(self, tfrecord_file, batchSize=128):
        self.tfrecord_file = tfrecord_file
        self.batchSize = batchSize
        self.curPos = 0

        self.dataset = tf.data.TFRecordDataset([self.tfrecord_file])

        self.dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        self.dataset = dataset.map(self._extract_fn)
        self.iterator = dataset.make_one_shot_iterator()
        self.next_image_data = iterator.get_next()


    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'rows': tf.FixedLenFeature([], tf.int64),
            'cols': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

        # Extract the data record
        sample = tf.parse_single_example(tfrecord, features)

        image = tf.image.decode_image(sample['image'])        
        img_shape = tf.stack([sample['rows'], sample['cols'], sample['channels']])
        label = sample['label'] = [0, 1] if sample['label'] == 0 else [1, 0]
        filename = sample['filename']
        return [image, label, filename, img_shape]  
    
    def NextTrainingBatch(self, tfSession):
        images_data = []

        for i in np.arange(self.batchSize):
            images_data.append(tfSession.run(self.next_image_data))

        return images_data


        
        
        



        
        
    