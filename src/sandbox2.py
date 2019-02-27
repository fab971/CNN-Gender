import numpy as np
import os
import time
from dataset.GenerateTFRecord import GenerateTFRecord
from glob import glob
from PIL import Image


PATH = '/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/CNN_Gender/data/resized/'

data_modes = ['train', 'test']
for d in data_modes:
    tfrecords_path = os.path.join(PATH, 'vggface2_dataset_{}.tfrecords'.format(d))
    tfBuilder = GenerateTFRecord()
    tfBuilder.convert_image_folder(PATH, d, tfrecords_path)


