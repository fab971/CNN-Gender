import tensorflow as tf
import os
import matplotlib.image as mpimg
from glob import glob
import csv

class GenerateTFRecord:
    def __init__(self):
        self.labels = []
        

    # data_mode = train | test
    def convert_image_folder(self, img_folder, data_mode, tfrecord_file_name):

        print('[{}] collecting data...'.format(data_mode))
        self.img_folder = img_folder
        self.data_mode = data_mode
        img_paths = []
        count = 0
        for root, dirs, files in os.walk(os.path.join(img_folder, data_mode)):
            count += 1
            img_paths.extend([f.replace(os.path.join(img_folder, data_mode) + '/', '') for f in glob(os.path.join(root, '*.jpg'))])
            print('================['+ str(count) +']================')
        
        img_paths.sort()

        print('[{}] collecting labels...'.format(data_mode))
        labels_dico = {}
        with open(os.path.join(img_folder, 'identity_meta.csv'), 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                seq, _, _, _, gender = row
                labels_dico[seq] = 0 if gender.strip() == 'm' else 1
        
        self.labels = [labels_dico[f.split('/')[0]] for f in img_paths]

        print('[{}] saving...'.format(data_mode))
        # Save
        with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
            for index, img_path in enumerate(img_paths):
                label = self.labels[index]
                example = self._convert_image(img_path, label)
                writer.write(example.SerializeToString())

    def _convert_image(self, img_path, label):
        img_path = os.path.join(self.img_folder, self.data_mode, img_path)
        img_shape = mpimg.imread(img_path).shape
        filename = os.path.basename(img_path)

        # Read image data in terms of bytes
        with tf.gfile.FastGFile(img_path, 'rb') as fid:
            image_data = fid.read()

        example = tf.train.Example(features = tf.train.Features(feature = {
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[2]])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
        }))
        return example