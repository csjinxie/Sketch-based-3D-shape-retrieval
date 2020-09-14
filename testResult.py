
# import system things
import tensorflow as tf
import numpy as np
import os
from dataset import Dataset
# from datetime import datetime
from scipy.spatial import distance
from RetrievalEvaluation import RetrievalEvaluation
import smtplib
# import helpers
import inference
# import visualize
from normData import normData
# for email
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('daiguoxian29@gmail.com', 'Dai29->Fool')

msg = 'Running is finished'

# prepare data
sketch_train_list = './sketchTrain.txt'
sketch_test_list = './sketchTest.txt'
shape_list = './shape.txt'
dataset = Dataset(sketch_train_list, sketch_test_list, shape_list)
C_depths = dataset.retrievalParamSP().astype(int)        # for retrieval evaluation
sketch_test_mean, sketch_test_std, sketch_train_mean, sketch_train_std, shape_mean, shape_std = dataset.normalizeData()
normLabel = 1
# setup siamese network
# net_type = 'metricAuto'                 # metricOnly: only use metric learning, metricAuto: using autoencoder with metric learning
net_type = 'holistic'
# net_type = 'metricOnly'
batch_train_size = 30
batch_test_size = 300
test_interval = 1
display_interval = 1
learning_rate = 0.1
momentum = 0.1
# resulTxt = net_type + '_noTanh.txt'
iterNum = 200000
resulTxt = net_type + str(iterNum) + '.txt'
preRecTxt = net_type + str(iterNum) + '_preRec.txt'
# resId = open(preRecTxt, 'w')
logid = open(resulTxt, 'w')
runOption = 0
# Construct Network ###############
siamese = inference.siamese(net_type=net_type)

# model_path = './models/' + net_type + '_noTanh_bal0001'# The reconstruction loss without tanh for output, the weight for reconstruction loss is 0.001
model_path = './models/' + net_type
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
if net_type == 'metricOnly':
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss)
    train_step_2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_2)
    train_step_1 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_1)
elif net_type == 'holistic':
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss)
    train_step_2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_2)
    train_step_1 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_1)

elif net_type == 'metricAuto':
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss)
    train_step_3 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.recon_error)
    train_step_2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_2)
    train_step_1 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(siamese.loss_1)
saver = tf.train.Saver(max_to_keep=10000)
init = tf.initialize_all_variables()
weights_file = os.path.join('./models', net_type, 'weight-' + str(iterNum))
with tf.Session() as sess:
    print('Initializing all variables')
    sess.run(init)
    if runOption == 1:          # starting from random weights
        print('Starting training from scratch*******')
    elif runOption == 0:        # restoring from previous weights
        print('Loading pretrained weight ****************')
        saver.restore(sess, weights_file)
    for step in range(1):
        # calculate sketch train fea
        if True:
            C_depths = dataset.retrievalParamSP().astype(int)        # for retrieval evaluation
            srcLabel = np.array(dataset.sketch_test_label).astype(float)
            dstLabel = np.array(dataset.shape_label).astype(float)
            backup_sketch_test_ptr = dataset.sketch_test_ptr       # backup for training
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
                    temp_fea = np.array(temp_fea)           # cast list into array
                    sketch_test_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea
                # calculate the left features
                remain_num = dataset.sketch_test_size - batch_num * batch_test_size
                temp_x1s, temp_y1s = dataset.next_batch(remain_num, 'sketch_test')
                if normLabel:
                    temp_x1s = normData(temp_x1s, sketch_test_mean, sketch_test_std)
                temp_fea = sess.run([siamese.o1], feed_dict={siamese.x1: temp_x1s})
                temp_fea = np.array(temp_fea)
                sketch_test_feaset[batch_num*batch_test_size:] = temp_fea
            # calculate shape fea
            backup_shape_ptr = dataset.shape_ptr       # backup for training
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
                    temp_fea = np.array(temp_fea)           # cast list into array
                    shape_feaset[i*batch_test_size: (i+1)*batch_test_size] = temp_fea
                # calculate the left features
                remain_num = dataset.shape_size - batch_num * batch_test_size
                temp_x1s, temp_y1s = dataset.next_batch(remain_num, 'shape')
                if normLabel:
                    temp_x1s = normData(temp_x1s, shape_mean, shape_std)
                temp_fea = sess.run([siamese.z1], feed_dict={siamese.y1: temp_x1s})
                temp_fea = np.array(temp_fea)
                shape_feaset[batch_num*batch_test_size:] = temp_fea
                dataset.shape_ptr = backup_shape_ptr         # restore pointer
            distM = distance.cdist(sketch_test_feaset, shape_feaset)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distM, dstLabel, srcLabel)
            pre = np.reshape(pre, (1, 8638))
            rec = np.reshape(rec, (1, 8638))
            preRec = np.concatenate((rec, pre), axis=0)
            np.savetxt(preRecTxt, preRec, fmt='%.5f')
            logid.write('The NN is {:.5f}\nThe FT is {:.5f}\nThe ST is {:.5f}\nThe DCG is {:.5f}\nThe E is {:.5f}\nThe MAP is {:.5f}'.format(nn_av, ft_av, st_av, dcg_av, e_av, map_))
            msg = 'SHREC 2013 Autometric Iteration {} \nThe NN is {:.5f}\nThe FT is {:.5f}\nThe ST is {:.5f}\nThe DCG is {:.5f}\nThe E is {:.5f}\nThe MAP is {:.5f}\n'.format(step, nn_av, ft_av, st_av, dcg_av, e_av, map_)
            print('SHREC 2013 Autometric Iteration {} \nThe NN is {:.5f}\nThe FT is {:.5f}\nThe ST is {:.5f}\nThe DCG is {:.5f}\nThe E is {:.5f}\nThe MAP is {:.5f}\n'.format(step, nn_av, ft_av, st_av, dcg_av, e_av, map_))
            # for email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login('daiguoxian29@gmail.com', 'Dai29->Fool')
            server.sendmail('daiguoxian29@gmail.com', 'daiguoxian29@gmail.com', msg)
            # print 'The NN is %5f' % (nn_av)
            # print 'The FT is %5f' % (ft_av)
            # print 'The ST is %5f' % (st_av)
            # print 'The DCG is %5f' % (dcg_av)
            # print 'The E is %5f' % (e_av)
            # print 'The MAP is %5f' % (map_)
    # email seeting ###########
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('daiguoxian29@gmail.com', 'Dai29->Fool')
    server.sendmail('daiguoxian29@gmail.com', 'daiguoxian29@gmail.com', 'SHREC 13 Running (10000 iterations) run.py is finished')
    server.quit()
    logid.close()
