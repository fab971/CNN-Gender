import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


class DataSets(object):
    def __init__(self, path, batchSize=128):
        self.path = path
        self.batchSize = batchSize
        self.curPos = 0

        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        batch_label.append(int(example.features.feature['gender'].int64_list.value[0]))
        filename = os.path.join(PATH, example.features.feature['path'].bytes_list.value[0].decode('utf-8'))

        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)

        batch_data.append(image_decoded)
    
    
#     def NextTrainingBatch(self):
# 		if self.curPos + self.batchSize > self.nbdata:
# 			self.curPos = 0
# 		xs = self.data[self.curPos:self.curPos+self.batchSize,:]
# 		ys = self.label[self.curPos:self.curPos+self.batchSize,:]
# 		self.curPos += self.batchSize
# 		return xs,ys






PATH = '/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/Gender_CNN/data/raw2/'


record_iterator = tf.python_io.tf_record_iterator(path=PATH + '../vggface2_dataset_10k.tfrecords')

count = 0

batch_label = []
batch_data = []

for string_record in record_iterator:
    
    count+=1

    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    batch_label.append(int(example.features.feature['gender'].int64_list.value[0]))
    filename = os.path.join(PATH, example.features.feature['path'].bytes_list.value[0].decode('utf-8'))

    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)

    batch_data.append(image_decoded)

    if count > 10:
        break
    
    # img_1d = np.fromstring(img_string, dtype=np.uint8)
    # reconstructed_img = img_1d.reshape((height, width, -1))
    
    # annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
    
    # # Annotations don't have depth (3rd dimension)
    # reconstructed_annotation = annotation_1d.reshape((height, width))
    
    # reconstructed_images.append((reconstructed_img, reconstructed_annotation))


sess = tf.Session()	
sess.run(tf.global_variables_initializer())


img = sess.run(batch_data)

print(batch_label)

plt.imshow(img[0])
plt.show()

print('done!')