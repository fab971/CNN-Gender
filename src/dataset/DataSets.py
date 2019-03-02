import numpy as np
import tensorflow as tf

class DataSet(object):

    def __init__(self, filepath_tfrecord, nbdata, L2normalize=False, batchSize=128):
        self.nbdata = nbdata
        self.batchSize = batchSize
        self.dim = 60 * 60 * 3
        self.curPos = 0
        self.data = None
        self.label = None

        self.dataset = tf.data.TFRecordDataset([filepath_tfrecord])
        self.dataset = self.dataset.map(self._extract_fn)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.next_image_data = self.iterator.get_next()

        self.L2normalize = L2normalize

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

    def load(self, TFsession):
        images_data = []

        for i in np.arange(self.nbdata):
            images_data.append(TFsession.run(self.next_image_data))
            
        np.random.shuffle(images_data)

        self.data = np.array([x[0] for x in images_data]).reshape(self.nbdata, self.dim)
        self.label = np.array([[0, 1] if x[1] == 0 else [1, 0] for x in images_data])


        tmpdata = np.empty([1, self.dim], dtype=np.float32)
        tmplabel = np.empty([1, 2], dtype=np.float32)
        arr = np.arange(self.nbdata)
        np.random.shuffle(arr)
        tmpdata = self.data[arr[0],:]
        tmplabel = self.label[arr[0],:]
        for i in range(self.nbdata-1):
            self.data[arr[i],:] = self.data[arr[i+1],:] 
            self.label[arr[i],:] = self.label[arr[i+1],:] 
        self.data[arr[self.nbdata-1],:] = tmpdata
        self.label[arr[self.nbdata-1],:] = tmplabel 

        if self.L2normalize:
            self.data /= np.sqrt(np.expand_dims(np.square(self.data).sum(axis=1), 1))

    def NextTrainingBatch(self):
        if self.curPos + self.batchSize > self.nbdata:
            self.curPos = 0
        xs = self.data[self.curPos:self.curPos+self.batchSize,:]
        ys = self.label[self.curPos:self.curPos+self.batchSize,:]
        self.curPos += self.batchSize
        return xs,ys
    
    def mean_accuracy(self, TFsession,loc_acc,loc_x,loc_y,loc_keep_prob):
        acc = 0
        for i in range(0, self.nbdata, self.batchSize):
            curBatchSize = min(self.batchSize, self.nbdata - i)
            dict = {loc_x:self.data[i:i+curBatchSize,:],loc_y:self.label[i:i+curBatchSize,:],loc_keep_prob:0.8}
            acc += TFsession.run(loc_acc, dict) * 	curBatchSize
        acc /= self.nbdata
        return acc
	