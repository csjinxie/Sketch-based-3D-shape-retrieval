""" Siamese implementation using Tensorflow
"""
#import system things
import tensorflow as tf
import numpy as np
import os
from dataset import Dataset
from datetime import datetime
from scipy.spatial import distance
from RetrievalEvaluation import RetrievalEvaluation
import smtplib
#import helpers
import inference
import visualize
from normData import normData
##### for email #####################
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('daiguoxian29@gmail.com', 'Dai29->Fool')

msg = 'Running is finished'

# prepare data
sketch_train_list = './sketchTrain.txt'
sketch_test_list = './sketchTest.txt'
shape_list = './shape.txt'
dataset = Dataset(sketch_train_list, sketch_test_list, shape_list)
C_depths = dataset.retrievalParamSP().astype(int);        ### for retrieval evaluation
sketch_test_mean, sketch_test_std, sketch_train_mean, sketch_train_std, shape_mean, shape_std = dataset.normalizeData()
normLabel = 1
# setup siamese network
batch_train_size = 30
batch_test_size = 300
test_interval = 500
display_interval = 3000
learning_rate = 0.1
momentum = 0.1
logid = open('temp.txt', 'w')
##################################################################################
siamese = inference.siamese();
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss)
train_step_2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_2)
train_step_1 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_1)
##############################################################################################################
saver = tf.train.Saver(max_to_keep=10000)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    print('Initializing all variables')
    sess.run(init)
    print('Restore training')
    # saver.restore(sess, 'models/models_99000.ckpt')
    saver.restore(sess, 'models/models_autometric_99000.ckpt')
    ########### calculate sketch train fea #######################################
    C_depths = dataset.retrievalParamSP().astype(int)        ### for retrieval evaluation
    srcLabel = np.array(dataset.sketch_test_label).astype(float)
    dstLabel = np.array(dataset.shape_label).astype(float)
    dataset.sketch_test_ptr = 0
    batch_num = dataset.sketch_test_size / batch_test_size
    batch_left = dataset.sketch_test_size % batch_test_size
    sketch_test_feaset = np.zeros((dataset.sketch_test_size, 100))
    if batch_left == 0:
        for i in range(batch_num):
            temp_x1s, temp_y1s = dataset.next_batch(batch_test_size, 'sketch_test')
            if normLabel:
                temp_x1s = normData(temp_x1s, sketch_test_mean, sketch_test_std)
            temp_fea = sess.run([siamese.o1], feed_dict={siamese.x1: temp_x1s})
            temp_fea = np.array(temp_fea)
            sketch_test_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea
    else:
        for i in range(batch_num):
            temp_x1s, temp_y1s = dataset.next_batch(batch_test_size, 'sketch_test')
            if normLabel:
                temp_x1s = normData(temp_x1s, sketch_test_mean, sketch_test_std)
            temp_fea = sess.run([siamese.o1], feed_dict={siamese.x1: temp_x1s})
            temp_fea = np.array(temp_fea)           ## cast list into array
            sketch_test_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea
                ### calculate the left features
        remain_num = dataset.sketch_test_size - batch_num * batch_test_size
        temp_x1s, temp_y1s = dataset.next_batch(remain_num, 'sketch_test')
        if normLabel:
            temp_x1s = normData(temp_x1s, sketch_test_mean, sketch_test_std)
        temp_fea = sess.run([siamese.o1], feed_dict={siamese.x1: temp_x1s})
        temp_fea = np.array(temp_fea)
        sketch_test_feaset[batch_num*batch_test_size:] = temp_fea
####### calculate shape fea #######################################################
### backup for training
    dataset.shape_ptr = 0
    batch_num = dataset.shape_size / batch_test_size
    batch_left = dataset.shape_size % batch_test_size
    shape_feaset = np.zeros((dataset.shape_size, 100))
    if batch_left == 0:
        for i in range(batch_num):
            temp_x1s, temp_y1s = dataset.next_batch(batch_test_size, 'shape')
            if normLabel:
                temp_x1s = normData(temp_x1s, shape_mean, shape_std)
            temp_fea = sess.run([siamese.z1], feed_dict={siamese.y1: temp_x1s})
            temp_fea = np.array(temp_fea)
            shape_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea
    else:
        for i in range(batch_num):
            temp_x1s, temp_y1s = dataset.next_batch(batch_test_size, 'shape')
            if normLabel:
                temp_x1s = normData(temp_x1s, shape_mean, shape_std)
            temp_fea = sess.run([siamese.z1], feed_dict={siamese.y1: temp_x1s})
            temp_fea = np.array(temp_fea)           ## cast list into array
            shape_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea
### calculate the left features
        remain_num = dataset.shape_size - batch_num * batch_test_size
        temp_x1s, temp_y1s = dataset.next_batch(remain_num, 'shape')
        if normLabel:
            temp_x1s = normData(temp_x1s, shape_mean, shape_std)
        temp_fea = sess.run([siamese.z1], feed_dict={siamese.y1: temp_x1s})
        temp_fea = np.array(temp_fea)
        shape_feaset[batch_num*batch_test_size:] = temp_fea
    distM = distance.cdist(sketch_test_feaset, shape_feaset)
    nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distM, dstLabel, srcLabel)
    print 'The NN is %5f' % (nn_av)
    print 'The FT is %5f' % (ft_av)
    print 'The ST is %5f' % (st_av)
    print 'The DCG is %5f' % (dcg_av)
    print 'The E is %5f' % (e_av)
    print 'The MAP is %5f' % (map_)
    server.sendmail('daiguoxian29@gmail.com', 'daiguoxian29@gmail.com', msg)
    server.quit()
    logid.close()

