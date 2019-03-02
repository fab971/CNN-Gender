import numpy as np
import os
import tensorflow as tf
import dataset.DataSets as ds
import matplotlib.pyplot as plt
import Layers

# test : 169,396 (500 identities)
# train : 3,141,890 (8631 identities)


LoadModel = False

train = ds.DataSet('/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/CNN_Gender/data/resized/vggface2_dataset_train.tfrecords', 100000)
test = ds.DataSet('/Users/fabienfluro/Documents/MS_BGD/Fil_Rouge/Work/CNN_Gender/data/resized/vggface2_dataset_test.tfrecords', 10000)

experiment_name = 'Gender prediction'

def get_dict(database):
	xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys,keep_prob:0.8}

with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, train.dim],name='x')
	y_desired = tf.placeholder(tf.float32, [None, 2],name='y_desired')
	keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('CNN'):
	t = Layers.unflat(x,60,60,3)
	t = Layers.conv(t,16,5,3,'conv_1')
	t = Layers.maxpool(t,5,'pool_2')
	t = Layers.batch_norm(t)
  
	t = Layers.conv(t,32,5,3,'conv_3')
	t = Layers.maxpool(t,5,'pool_4')
	t = Layers.batch_norm(t)
  
	t = Layers.conv(t,128,5,3,'conv_5')
	t = Layers.maxpool(t,5,'pool_6')
	t = Layers.batch_norm(t)
  
	t = Layers.flat(t)
  
	t = Layers.fc(t,1024,'fc_7')
	t = tf.nn.dropout(t, keep_prob)

	t = Layers.fc(t,128,'fc_8')
	t = tf.nn.dropout(t, keep_prob)
  
	y = Layers.fc(t,2,'fc_9',tf.nn.log_softmax)



with tf.name_scope('cross_entropy'):
	diff = y_desired * y
	with tf.name_scope('total'):
		cross_entropy = -tf.reduce_mean(diff)
	tf.summary.scalar('cross entropy', cross_entropy)	
	
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_desired, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)	

with tf.name_scope('learning_rate'):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(0.001,global_step,1000, 0.75, staircase=True)


with tf.name_scope('learning_rate'):
    tf.summary.scalar('learning_rate', learning_rate)


train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
merged = tf.summary.merge_all() #

Acc_Train = tf.placeholder("float", name='Acc_Train')
Acc_Test = tf.placeholder("float", name='Acc_Test')
MeanAcc_summary = tf.summary.merge([tf.summary.scalar('Acc_Train', Acc_Train),tf.summary.scalar('Acc_Test', Acc_Test)])


print ("-----------------------------------------------------")
print ("-----------",experiment_name)
print ("-----------------------------------------------------")


sess = tf.Session()	
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(experiment_name, sess.graph)
saver = tf.train.Saver()
if LoadModel:
	saver.restore(sess, "./model.ckpt")


train.load(sess)
test.load(sess)

nbIt = 1000
for it in range(nbIt):
	trainDict = get_dict(train)
					
	sess.run(train_step, feed_dict=trainDict)
	if it%10 == 0:
		acc,ce = sess.run([accuracy,cross_entropy], feed_dict=trainDict)
		print ("it= %6d - cross_entropy= %f - acc= %f" % (it,ce,acc ))
		summary_merged = sess.run(merged, feed_dict=trainDict)
		writer.add_summary(summary_merged, it)	
		
	if it%100 == 50:
		Acc_Train_value = train.mean_accuracy(sess,accuracy,x,y_desired,keep_prob)
		Acc_Test_value = test.mean_accuracy(sess,accuracy,x,y_desired,keep_prob)
		print ("mean accuracy train = %f  test = %f" % (Acc_Train_value,Acc_Test_value ))
		summary_acc = sess.run(MeanAcc_summary, feed_dict={Acc_Train:Acc_Train_value,Acc_Test:Acc_Test_value})
		writer.add_summary(summary_acc, it)
		if Acc_Test_value > 0.9 and not LoadModel:
			# saver.save(sess, "/content/gdrive/My Drive/Colab Notebooks/CNN_GENDER/model/model.ckpt")
			saver.save(sess, "./model.ckpt")
		
writer.close()
# if not LoadModel:
# 	saver.save(sess, "./model.ckpt")
sess.close()















# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# data, labels = train.NextTrainingBatch(sess)


# sess.close()

# print(data)
# print(labels)


# print(np.shape(images_data[0][0]))
# print('data:')
# print(images_data[0])
# print('labels:')
# print(images_data[1])


# print(np.shape(images_data))

# plt.imshow(data[0])
# plt.show()
print('done!')