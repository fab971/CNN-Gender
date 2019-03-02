import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


class DataSet(object):
    def __init__(self, filepath_tfrecord, nbdata, batchSize=128):
        self.nbdata = nbdata
        self.batchSize = batchSize
        self.dim = 60 * 60 * 3
        self.curPos = 0

        self.dataset = tf.data.TFRecordDataset([filepath_tfrecord])
        self.dataset = self.dataset.map(self._extract_fn)
        self.dataset = self.dataset.shuffle(buffer_size=20000)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.next_image_data = self.iterator.get_next()

        self.iterator0 = self.dataset.make_one_shot_iterator()
        self.next_image_data0 = self.iterator0.get_next()

        self.currentDataBatch = []
        self.currentLabelBatch = []

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
        img_shape = tf.stack(
            [sample['rows'], sample['cols'], sample['channels']])
        label = sample['label']
        return [image, label, img_shape]

    def NextTrainingBatch(self, TFsession):
        images_data = []

        for i in np.arange(self.batchSize):
            images_data.append(TFsession.run(self.next_image_data))

        self.currentDataBatch = np.array(
            [x[0] for x in images_data]).reshape(self.batchSize, self.dim)
        self.currentLabelBatch = np.array(
            [[0, 1] if x[1] == 0 else [1, 0] for x in images_data])

        return self.currentDataBatch, self.currentLabelBatch

    def mean_accuracy(self, TFsession, loc_acc, loc_x, loc_y):
        acc = 0
        images_data0 = []
        for i in range(0, self.nbdata, self.batchSize):
            curBatchSize = min(self.batchSize, self.nbdata - i)
            
            for i in np.arange(curBatchSize):
                images_data0.append(TFsession.run(self.next_image_data0))

            
            currentDataBatch0 = np.array([x[0] for x in images_data0]).reshape(self.nbdata, self.dim)
            currentLabelBatch0 = np.array([[0, 1] if x[1] == 0 else [1, 0] for x in images_data0])
            
            dict = {loc_x:currentDataBatch0,loc_y:currentLabelBatch0}
            acc += TFsession.run(loc_acc, dict) * 	curBatchSize
            
        acc /= self.nbdata
        return acc
