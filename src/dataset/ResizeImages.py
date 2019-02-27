# import tensorflow as tf
import numpy as np
from PIL import Image
from resizeimage import resizeimage
import os
from glob import glob

base_path = '/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/CNN_Gender/data'
base_path_resized = '/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/CNN_Gender/data/resized'

resized_width = 60
resized_height = 60

if not os.path.exists(base_path_resized):
    os.makedirs(base_path_resized)

data_type = ['train', 'test']
# data_type = ['sandbox']

for d in data_type:
    print('[{}] collecting data...'.format(d))
    filenames = []
    count = 0
    for root, dirs, files in os.walk(os.path.join(base_path, d)):
        count += 1
        file_paths = [f for f in glob(os.path.join(root, '*.jpg'))]
        for file_path in file_paths:
            with open(file_path, 'r+b') as f:
                with Image.open(f) as image:
                    relative_filename = file_path.replace(base_path, '')
                    try:
                        cover = resizeimage.resize_cover(image, [resized_width, resized_height])
                        file_path_resized = base_path_resized + relative_filename
                        dir_resized = os.path.dirname(file_path_resized)
                        if not os.path.exists(dir_resized):
                            os.makedirs(dir_resized)
                        cover.save(file_path_resized, image.format)
                    except:
                        pass
                    


        print('================['+ str(count) +']================')


print('done')