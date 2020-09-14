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

#import helpers
import inference
import visualize
from normData import normData

# prepare data
sketch_train_list = './sketchTrain.txt'
sketch_test_list = './sketchTest.txt'
shape_list = './shape.txt'
dataset = Dataset(sketch_train_list, sketch_test_list, shape_list)
C_depths = dataset.retrievalParamSS().astype(int);        ### for retrieval evaluation
sketch_test_mean, sketch_test_std, sketch_train_mean, sketch_train_std, shape_mean, shape_std = dataset.normalizeData()
normLabel = 1
org_test = 0
# setup siamese network
batch_train_size = 30
batch_test_size = 300
test_interval = 100
display_interval = 3000
learning_rate = 0.01
momentum = 0.1
logid = open('myLog.txt', 'w')
##################################################################################
siamese = inference.siamese();
# For both network
train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss)
# For the second network
train_step_2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_2)
#train_step_2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(siamese.loss_2)

# For the third network
train_step_1 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_1)
#train_step_1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(siamese.loss_1)
##############################################################################################################
saver = tf.train.Saver(max_to_keep=10000)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    print('Initializing all variables')
    sess.run(init)
    print('Starting training')
    for step in range(100000):
        ########### for sketch ##################################
        sketch_x1s, sketch_y1s = dataset.next_batch(batch_train_size, 'shape')           #### first sketch batch
        sketch_x2s, sketch_y2s = dataset.next_batch(batch_train_size, 'shape')           #### second sketch batch
        if normLabel:
            ######## for sketch ############################
            sketch_x1s = normData(sketch_x1s, shape_mean, shape_std)
            sketch_x2s = normData(sketch_x2s, shape_mean, shape_std)
        label2 = (sketch_y1s == sketch_y2s).astype(int)                           #### label similarity for sketch
        label1 = np.zeros(batch_train_size)                             #### label similarity for shape
        label3 = np.zeros(batch_train_size)                            #### label similariy for sketch-shape
        #print(label1)
        label = np.array([label1, label2, label3]).astype(float)                    #### assemble label array
        label = np.transpose(label, (1, 0))
        _, loss_v = sess.run([train_step_2, siamese.loss_2], feed_dict={
                            siamese.y1: sketch_x1s,
                            siamese.y2: sketch_x2s,
                            siamese.simLabel: label})
############################################################################################
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()
        if step % test_interval == 0:
            print ('{} Iter {:5d}: loss {:.3f}'.format(datetime.now(), step, loss_v))
            ########### calculate sketch train fea #######################################
        if step % display_interval == 0:
            ##########  retrieval data #############
            C_depths = dataset.retrievalParamPP().astype(int)        ### for retrieval evaluation
            srcLabel = np.array(dataset.shape_label).astype(float)
            dstLabel = np.array(dataset.shape_label).astype(float)
            #########################################################################
            model_path = './models_' + str(step)+'.ckpt'
            saver.save(sess, model_path)
            backup_shape_ptr = dataset.shape_ptr       ### backup for training
            dataset.shape_ptr = 0
            batch_num = dataset.shape_size / batch_test_size
            batch_left = dataset.shape_size % batch_test_size
            if org_test == 1:
                shape_feaset = np.ndarray((dataset.shape_size, 4096))
            else:
                shape_feaset = np.ndarray((dataset.shape_size, 100))
            if batch_left == 0:
                for i in range(batch_num):
                    temp_x1s, temp_y1s = dataset.next_batch(batch_test_size, 'shape')
                    if normLabel:
                        temp_x1s = normData(temp_x1s, shape_mean, shape_std)
                    if org_test == 1:
                        temp_fea = temp_x1s
                    else:
                        temp_fea = sess.run([siamese.z1], feed_dict={siamese.y1: temp_x1s})
                    temp_fea = np.array(temp_fea)
                    shape_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea
            else:
                for i in range(batch_num):
                    temp_x1s, temp_y1s = dataset.next_batch(batch_test_size, 'shape')
                    if normLabel:
                        temp_x1s = normData(temp_x1s, shape_mean, shape_std)
                    if org_test == 1:
                        temp_fea = temp_x1s
                    else:
                        temp_fea = sess.run([siamese.z1], feed_dict={siamese.y1: temp_x1s})
                    temp_fea = np.array(temp_fea)           ## cast list into array
                    shape_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea
                remain_num = dataset.shape_size - batch_num * batch_test_size
                temp_x1s, temp_y1s = dataset.next_batch(remain_num, 'shape')
                if normLabel:
                    temp_x1s = normData(temp_x1s, shape_mean, shape_std)
                if org_test == 1:
                    temp_fea = temp_x1s
                else:
                    temp_fea = sess.run([siamese.z1], feed_dict={siamese.y1: temp_x1s})
                temp_fea = np.array(temp_fea)
                shape_feaset[batch_num*batch_test_size:] = temp_fea
            distM = distance.cdist(shape_feaset, shape_feaset)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distM, dstLabel, srcLabel)
            print 'The NN is %5f' % (nn_av)
            print 'The FT is %5f' % (ft_av)
            print 'The ST is %5f' % (st_av)
            print 'The DCG is %5f' % (dcg_av)
            print 'The E is %5f' % (e_av)
            print 'The MAP is %5f' % (map_)
    logid.closed()


