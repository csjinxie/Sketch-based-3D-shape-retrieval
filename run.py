""" Siamese implementation using Tensorflow
"""
#import system things
import tensorflow as tf
import numpy as np
from utils import *
import os
from dataset import Dataset
from datetime import datetime
from scipy.spatial import distance
from RetrievalEvaluation import RetrievalEvaluation
import smtplib
#import helpers
import inference
# import inference_old
import visualize
from normData import normData
import sys

import argparse
##### for email #####################

tf.set_random_seed(22222)
np.random.seed(22222)



# parameter initialization

parser = argparse.ArgumentParser(description='This is for cross domain metric learning')

parser.add_argument('--sketch_train_list', type=str, default='./sketchTrain.txt')
parser.add_argument('--sketch_test_list', type=str, default='./sketchTest.txt')
parser.add_argument('--shape_list', type=str, default='./shape.txt')
parser.add_argument('--inputFeaSize', type=int, default=4096)
parser.add_argument('--net_type', type=str, default='metricOnly')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--batch_train_size', type=int, default=30)
parser.add_argument('--batch_test_size', type=int, default=300)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--display_interval', type=int, default=500)
parser.add_argument('--logfile', type=str, default='log.txt')
parser.add_argument('--normLabel', type=int, default=1)
parser.add_argument('--maxIter', type=int, default=200000)
parser.add_argument('--margin_out', type=float, default=3.)
parser.add_argument('--margin_hid', type=float, default=40.)
parser.add_argument('--alpha', type=float, default=0.3)                             # alpha is used to balance the hidden layer correlation loss and output layer correlation loss
parser.add_argument('--class_num', type=int, default=171)                             # alpha is used to balance the hidden layer correlation loss and output layer correlation loss




args = parser.parse_args()

print("{:20}\t = {:20}".format('sketch_train_list', args.sketch_train_list))
print("{:20}\t = {:20}".format('sketch_test_list', args.sketch_test_list))
print("{:20}\t = {:20}".format('shape_list', args.shape_list))
print("{:20}\t = {:20}".format('inputFeaSize', args.inputFeaSize))
print("{:20}\t = {:20}".format('net_type', args.net_type))
print("{:20}\t = {:20}".format('learning_rate', args.learning_rate))
print("{:20}\t = {:20}".format('momentum', args.momentum))
print("{:20}\t = {:20}".format('batch_train_size', args.batch_train_size))
print("{:20}\t = {:20}".format('batch_test_size', args.batch_test_size))
print("{:20}\t = {:20}".format('test_interval', args.test_interval))
print("{:20}\t = {:20}".format('display_interval', args.display_interval))
print("{:20}\t = {:20}".format('logfile', args.logfile))
print("{:20}\t = {:20}".format('normLabel', args.normLabel))
print("{:20}\t = {:20}".format('maxIter', args.maxIter))
print("{:20}\t = {:20}".format('margin_out', args.margin_out))
print("{:20}\t = {:20}".format('margin_hid', args.margin_hid))
print("{:20}\t = {:20}".format('alpha', args.alpha))
print("{:20}\t = {:20}".format('clss_num', args.class_num))

# prepare data

dataset = Dataset(sketch_train_list=args.sketch_train_list, sketch_test_list=args.sketch_test_list, shape_list=args.shape_list, inputFeaSize=args.inputFeaSize, class_num=args.class_num)
C_depths = dataset.retrievalParamSP().astype(int);        ### for retrieval evaluation
sketch_test_mean, sketch_test_std, sketch_train_mean, sketch_train_std, shape_mean, shape_std = dataset.normalizeData()
# setup siamese network
batch_train_size = args.batch_train_size
batch_test_size = args.batch_test_size
test_interval = args.test_interval
display_interval = args.display_interval

logid = open(args.logfile, 'w')

logid.write("Training results for {:10}".format(args.net_type))
logid.close()

logid = open(args.logfile, 'a')



checkpoint_dir = os.path.join('../models', args.net_type)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

modelName = 'model'
ckpt_status = get_checkpoint(checkpoint_dir)

##################################################################################
siamese = inference.siamese(net_type=args.net_type, margin_out=args.margin_out, margin_hid=args.margin_hid, alpha=args.alpha) # net_type: chosing metric only or with holistic loss
# siamese = inference_old.siamese()


train_step = tf.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=args.momentum).minimize(siamese.loss)
# train_step_2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_2)
# train_step_1 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_1)
##############################################################################################################

saver = tf.train.Saver(max_to_keep=10000)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    print('Initializing all variables')
    sess.run(init)
    if ckpt_status:
        print("Successfully loading the checkpoint")
        saver.restore(sess, ckpt_status)
    else:
        print("Failed to load checkpoint")

    for step in range(args.maxIter):
        # Loading sketch feature
        sketch_x1s, sketch_y1s = dataset.next_batch(batch_train_size, 'sketch_train')           #### first sketch batch
        sketch_x2s, sketch_y2s = dataset.next_batch(batch_train_size, 'sketch_train')           #### second sketch batch

        # Loading shape feature
        shape_x1s, shape_y1s = dataset.next_batch(batch_train_size, 'shape')              #### first shape batch
        shape_x2s, shape_y2s = dataset.next_batch(batch_train_size, 'shape')              #### second shape batch

        # Normalize the input features
        if args.normLabel:
            # normalize sketch feature
            sketch_x1s = normData(sketch_x1s, sketch_train_mean, sketch_train_std)
            sketch_x2s = normData(sketch_x2s, sketch_train_mean, sketch_train_std)

            # normalize shape feature
            shape_x1s = normData(shape_x1s, shape_mean, shape_std)
            shape_x2s = normData(shape_x2s, shape_mean, shape_std)

        # optimize the network

        _, loss_all, loss_sketch, loss_shape, loss_cross = sess.run([train_step, siamese.loss, siamese.loss_1, siamese.loss_2, siamese.loss_3], feed_dict={siamese.input_sketch_fea_1: sketch_x1s,
            siamese.input_sketch_label_1: sketch_y1s,
            siamese.input_sketch_fea_2: sketch_x2s,
            siamese.input_sketch_label_2: sketch_y2s,
            siamese.input_shape_fea_1: shape_x1s,
            siamese.input_shape_label_1: shape_y1s,
            siamese.input_shape_fea_2: shape_x2s,
            siamese.input_shape_label_2: shape_y2s})

        # IF the loss reaches NAN, stop training
        if np.isnan(loss_all):
            print('Model diverged with loss = NaN')
            quit()

        if step % display_interval == 0:
            #print ('{} Iter {:5d}: loss {:.3f}, loss_1 {:.3f}, loss_2 {:.3f}'.format(datetime.now(), step, loss_v, loss_v1, loss_v2))
            print ('{} Iter {:5d}: loss\t{:.3f}\t\t loss_sketch\t{:.3f} \t\tloss_shape\t{:.3f} \t\tloss_cross\t{:.3f}'.format(datetime.now(), step, loss_all, loss_sketch, loss_shape, loss_cross))
            logid.write('{} Iter {:5d}: loss {:.3f}\n'.format(datetime.now(), step, loss_all))

            ########### calculate sketch train fea #######################################
        if step % test_interval == 0 and step > 0:

            # Get retrieval input labels
            C_depths = dataset.retrievalParamSP().astype(int)        ### for retrieval evaluation
            #srcLabel = np.array(dataset.sketch_test_label).astype(float)
            #dstLabel = np.array(dataset.shape_label).astype(float)
            srcLabel = dataset.sketchTestLabelset
            dstLabel = dataset.shapeLabelset

            model_path = os.path.join(checkpoint_dir, modelName)
            saver.save(sess, model_path, global_step=step)

            print("Loading test sketch features")
            batch_num = int(dataset.sketch_test_size / batch_test_size)
            batch_left = dataset.sketch_test_size % batch_test_size
            sketch_test_feaset = np.zeros((dataset.sketch_test_size, 100))
            for i in range(batch_num):
                temp_x1s = dataset.sketchTestFeaset[i*batch_test_size:(i+1)*batch_test_size]
                if args.normLabel:
                    temp_x1s = normData(temp_x1s, sketch_test_mean, sketch_test_std)
                temp_fea = sess.run(siamese.o1, feed_dict={siamese.input_sketch_fea_1: temp_x1s})
                sketch_test_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea

            if batch_left != 0:
                temp_x1s = dataset.sketchTestFeaset[batch_num*batch_test_size:]
                if args.normLabel:
                    temp_x1s = normData(temp_x1s, sketch_test_mean, sketch_test_std)
                temp_fea = sess.run(siamese.o1, feed_dict={siamese.input_sketch_fea_1: temp_x1s})
                sketch_test_feaset[batch_num*batch_test_size:] = temp_fea


            print("Loading shape features")
            batch_num = int(dataset.shape_size / batch_test_size)
            batch_left = dataset.shape_size % batch_test_size
            shape_feaset = np.zeros((dataset.shape_size, 100))
            for i in range(batch_num):
                temp_x1s = dataset.shapeFeaset[i*batch_test_size:(i+1)*batch_test_size]
                if args.normLabel:
                    temp_x1s = normData(temp_x1s, shape_mean, shape_std)
                temp_fea = sess.run(siamese.z1, feed_dict={siamese.input_shape_fea_1: temp_x1s})
                shape_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea


            if batch_left != 0:
                temp_x1s = dataset.shapeFeaset[batch_num*batch_test_size:]
                if args.normLabel:
                    temp_x1s = normData(temp_x1s, sketch_test_mean, sketch_test_std)
                temp_fea = sess.run(siamese.z1, feed_dict={siamese.input_shape_fea_1: temp_x1s})
                shape_feaset[batch_num*batch_test_size:] = temp_fea


            distM = distance.cdist(sketch_test_feaset, shape_feaset)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distM, dstLabel, srcLabel)
            print('The NN is {:.5f}\nThe FT is {:.5f}\nThe ST is {:.5f}\nThe DCG is {:.5f}\nThe E is {:.5f}\nThe MAP is {:.5f}'.format(nn_av, ft_av, st_av, dcg_av, e_av, map_))
            logid.write('The NN is {:.5f}\nThe FT is {:.5f}\nThe ST is {:.5f}\nThe DCG is {:.5f}\nThe E is {:.5f}\nThe MAP is {:.5f}\n'.format(nn_av, ft_av, st_av, dcg_av, e_av, map_))

    logid.close()
