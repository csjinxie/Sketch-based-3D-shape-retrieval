
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
#net_type = 'metricAuto'                 ## metricOnly: only use metric learning, metricAuto: using autoencoder with metric learning
net_type = 'holistic'
# net_type = 'metricOnly'
batch_train_size = 30
batch_test_size = 300
test_interval = 500
display_interval = 5000
learning_rate = 0.1
momentum = 0.1
# #resulTxt = net_type + '_noTanh.txt'
# resulTxt = net_type + 'Appeend.txt'
resulTxt = net_type + '.txt'
logid = open(resulTxt, 'w')
runOption = 1
##Construct Network ###############
siamese = inference.siamese(net_type=net_type)
#model_path = './models/' + net_type + '_noTanh_bal0001'# The reconstruction loss without tanh for output, the weight for reconstruction loss is 0.001
model_path = os.path.join('./models', net_type, 'weight')
weights_file = './models/metricOnly-195000'
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
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
##############################################################################################################
saver = tf.train.Saver(max_to_keep=10000)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    print('Initializing all variables')
    sess.run(init)
    if runOption == 1: ### starting from random weights
        print('Starting training from scratch*******')
    elif runOption == 0: ### restoring from previous weights
        print('Loading pretrained weight ****************')
        saver.restore(sess, weights_file)
    for step in range(250000):
        ########### for sketch ##################################
        sketch_x1s, sketch_y1s = dataset.next_batch(batch_train_size, 'sketch_train')           #### first sketch batch
        sketch_x2s, sketch_y2s = dataset.next_batch(batch_train_size, 'sketch_train')           #### second sketch batch
        ########## for shape ######################################
        shape_x1s, shape_y1s = dataset.next_batch(batch_train_size, 'shape')              #### first shape batch
        shape_x2s, shape_y2s = dataset.next_batch(batch_train_size, 'shape')              #### second shape batch
        if normLabel:
            ######## for sketch ############################
            sketch_x1s = normData(sketch_x1s, sketch_train_mean, sketch_train_std)
            sketch_x2s = normData(sketch_x2s, sketch_train_mean, sketch_train_std)
            ######## for shape #############################
            shape_x1s = normData(shape_x1s, shape_mean, shape_std)
            shape_x2s = normData(shape_x2s, shape_mean, shape_std)
        label1 = (sketch_y1s == sketch_y2s).astype(float)                           #### label similarity for sketch
        label2 = (shape_y1s == shape_y2s).astype(float)                             #### label similarity for shape
        label3 = (sketch_y1s == shape_y1s).astype(float)                            #### label similariy for sketch-shape
        #print(label1)
        label = np.array([label1, label2, label3]).astype(float)                    #### assemble label array
        label = np.transpose(label, (1, 0))
        if step > -1:
            _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: sketch_x1s,
                            siamese.x2: sketch_x2s,
                            siamese.y1: shape_x1s,
                            siamese.y2: shape_x2s,
                            siamese.simLabel: label})
################### add more training for shape #########################################

#             for i in range(5):
#                 label1 = (sketch_y1s == sketch_y2s).astype(float)                           #### label similarity for sketch
#                 shape_x3s, shape_y3s = dataset.next_batch(batch_train_size, 'shape')              #### third shape batch
#                 shape_x4s, shape_y4s = dataset.next_batch(batch_train_size, 'shape')              #### forth shape batch
#                 if normLabel:
#             ######## for shape #############################
#                     shape_x3s = normData(shape_x3s, shape_mean, shape_std)
#                     shape_x4s = normData(shape_x4s, shape_mean, shape_std)
# #############################################################################################################################
#                 label2 = (shape_y3s == shape_y4s).astype(float)                             #### label similarity for shape
#                 label3 = (sketch_y1s == shape_y1s).astype(float)                            #### label similariy for sketch-shape
#                 label = np.array([label1, label2, label3]).astype(float)                    #### assemble label array
#                 label = np.transpose(label, (1, 0))
#                 _, loss_v2 = sess.run([train_step_2, siamese.loss_2], feed_dict={
#                             siamese.y1: shape_x3s,
#                             siamese.y2: shape_x4s,
#                             siamese.simLabel: label})
# ################### add more training for sketch #########################################
#             sketch_x3s, sketch_y3s = dataset.next_batch(batch_train_size, 'sketch_train')           #### first sketch batch
#             sketch_x4s, sketch_y4s = dataset.next_batch(batch_train_size, 'sketch_train')           #### second sketch batch
#             if normLabel:
#             ######## for sketch ############################
#                 sketch_x3s = normData(sketch_x3s, sketch_train_mean, sketch_train_std)
#                 sketch_x4s = normData(sketch_x4s, sketch_train_mean, sketch_train_std)


#             label1 = (sketch_y3s == sketch_y4s).astype(float)                           #### label similarity for sketch
#             label2 = (shape_y3s == shape_y4s).astype(float)                             #### label similarity for shape
#             label3 = (sketch_y1s == shape_y1s).astype(float)                            #### label similariy for sketch-shape
#             label = np.array([label1, label2, label3]).astype(float)                    #### assemble label array
#             label = np.transpose(label, (1, 0))
#             _, loss_v1 = sess.run([train_step_1, siamese.loss_1], feed_dict={
#                             siamese.x1: sketch_x3s,
#                             siamese.x2: sketch_x4s,
#                             siamese.simLabel: label})
#############################################################################################
        else:
            _, loss_v = sess.run([train_step_3, siamese.recon_error], feed_dict={
                            siamese.x1: sketch_x1s,
                            siamese.x2: sketch_x2s,
                            siamese.y1: shape_x1s,
                            siamese.y2: shape_x2s,
                            siamese.simLabel: label})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()
        if step % test_interval == 0:
            #print ('{} Iter {:5d}: loss {:.3f}, loss_1 {:.3f}, loss_2 {:.3f}'.format(datetime.now(), step, loss_v, loss_v1, loss_v2))
            print ('{} Iter {:5d}: loss {:.3f}'.format(datetime.now(), step, loss_v))
            logid.write('{} Iter {:5d}: loss {:.3f}\n'.format(datetime.now(), step, loss_v))

            ########### calculate sketch train fea #######################################
        if step % display_interval == 0 and step > 0:
            C_depths = dataset.retrievalParamSP().astype(int)        ### for retrieval evaluation
            srcLabel = np.array(dataset.sketch_test_label).astype(float)
            dstLabel = np.array(dataset.shape_label).astype(float)

            saver.save(sess, model_path, global_step=step)
            backup_sketch_test_ptr = dataset.sketch_test_ptr       ### backup for training
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
            ####### calculate shape fea ########################################################
            backup_shape_ptr = dataset.shape_ptr       ### backup for training
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
                dataset.shape_ptr = backup_shape_ptr         ## restore pointer
            distM = distance.cdist(sketch_test_feaset, shape_feaset)
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distM, dstLabel, srcLabel)
            logid.write('The NN is {:.5f}\nThe FT is {:.5f}\nThe ST is {:.5f}\nThe DCG is {:.5f}\nThe E is {:.5f}\nThe MAP is {:.5f}'.format(nn_av, ft_av, st_av, dcg_av, e_av, map_))
            msg = 'SHREC 2013 Autometric Iteration {} \nThe NN is {:.5f}\nThe FT is {:.5f}\nThe ST is {:.5f}\nThe DCG is {:.5f}\nThe E is {:.5f}\nThe MAP is {:.5f}\n'.format(step, nn_av, ft_av, st_av, dcg_av, e_av, map_)
            print('SHREC 2013 Autometric Iteration {} \nThe NN is {:.5f}\nThe FT is {:.5f}\nThe ST is {:.5f}\nThe DCG is {:.5f}\nThe E is {:.5f}\nThe MAP is {:.5f}\n'.format(step, nn_av, ft_av, st_av, dcg_av, e_av, map_))
############ for email #####################
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
    ## email seeting ###########
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('daiguoxian29@gmail.com', 'Dai29->Fool')
    server.sendmail('daiguoxian29@gmail.com', 'daiguoxian29@gmail.com', 'SHREC 13 Running (10000 iterations) run.py is finished')
    server.quit()
    logid.close()

