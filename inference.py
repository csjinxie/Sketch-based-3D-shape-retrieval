import tensorflow as tf
import sys

class siamese:

    # Create model
    def __init__(self, margin_out=3.0, margin_hid=40., alpha=0.1, net_type='metricOnly'):
        """
        initialize all variables
        """

        self.margin_out = margin_out
        self.margin_hid = margin_hid
        self.alpha      = alpha             # alpha is used to balance the hidden layer correlation loss and output layer correlation loss

        # alpha * out_layer + (1. - alpha) * hidden_layer



        # sketch feature place holder
        self.input_sketch_fea_1 = tf.placeholder(tf.float32, [None, 4096], name='input_sketch_fea_1')           ### for sktech 1
        self.input_sketch_fea_2 = tf.placeholder(tf.float32, [None, 4096], name='input_sketch_fea_2')           ### for sketch 2

        # shape feature place holder
        self.input_shape_fea_1  = tf.placeholder(tf.float32, [None, 4096], name='input_shape_fea_1')           ### for shape 1
        self.input_shape_fea_2  = tf.placeholder(tf.float32, [None, 4096], name='input_shape_fea_2')           ### for shape 2


        # sketch label placehoder
        self.input_sketch_label_1 = tf.placeholder(tf.float32, [None, 1], name='input_sketch_label_1')           ### for sktech 1
        self.input_sketch_label_2 = tf.placeholder(tf.float32, [None, 1], name='input_sketch_label_2')           ### for sketch 2

        # shape label placeholder
        self.input_shape_label_1  = tf.placeholder(tf.float32, [None, 1], name='input_shape_label_1')           ### for shape 1
        self.input_shape_label_2  = tf.placeholder(tf.float32, [None, 1], name='input_shape_label_2')           ### for shape 2



        # Generate similarity label
        self.sketch_simLabel            =   self.genSimLabel(self.input_sketch_label_1, self.input_sketch_label_2)
        self.shape_simLabel             =   self.genSimLabel(self.input_shape_label_1, self.input_shape_label_2)
        self.sketch_shape_simLabel_1    =   self.genSimLabel(self.input_sketch_label_1, self.input_shape_label_1)
        self.sketch_shape_simLabel_2    =   self.genSimLabel(self.input_sketch_label_2, self.input_shape_label_2)

        if net_type == 'metricOnly': ## only use deep metric
            with tf.variable_scope("sketch") as scope:
                self.o1 = self.sketchNetwork(self.input_sketch_fea_1)
                scope.reuse_variables()
                self.o2 = self.sketchNetwork(self.input_sketch_fea_2)
            with tf.variable_scope("shape") as scope:
                self.z1 = self.shapeNetwork(self.input_shape_fea_1)
                scope.reuse_variables()
                self.z2 = self.shapeNetwork(self.input_shape_fea_2)
            # Create loss
            self.loss,      self.loss_1,        self.loss_2,        self.loss_3         = self.contrastiveLoss()

        elif net_type == 'shareWeight': ## only use deep metric
            with tf.variable_scope("sketch") as scope:
                self.o1 = self.sketchNetwork(self.input_sketch_fea_1)
                scope.reuse_variables()
                self.o2 = self.sketchNetwork(self.input_sketch_fea_2)
                scope.reuse_variables()
                self.z1 = self.sketchNetwork(self.input_shape_fea_1)
                scope.reuse_variables()
                self.z2 = self.sketchNetwork(self.input_shape_fea_2)
            # Create loss
            self.loss,      self.loss_1,        self.loss_2,        self.loss_3         = self.contrastiveLoss()

        elif net_type == 'holistic': ## only use deep metric
            with tf.variable_scope("sketch") as scope:
                self.o1, self.ho1 = self.sketchNetworkHolistic(self.input_sketch_fea_1)
                scope.reuse_variables()
                self.o2, self.ho2 = self.sketchNetworkHolistic(self.input_sketch_fea_2)
            with tf.variable_scope("shape") as scope:
                self.z1, self.hz1 = self.shapeNetworkHolistic(self.input_shape_fea_1)
                scope.reuse_variables()
                self.z2, self.hz2 = self.shapeNetworkHolistic(self.input_shape_fea_2)

            # Create loss
            self.loss,      self.loss_1,        self.loss_2,        self.loss_3         = self.contrastiveLossHolistic()

        elif net_type == 'alpha': ## only use deep metric
            with tf.variable_scope("sketch") as scope:
                self.o1, self.ho1 = self.sketchNetworkHolistic(self.input_sketch_fea_1)
                scope.reuse_variables()
                self.o2, self.ho2 = self.sketchNetworkHolistic(self.input_sketch_fea_2)
            with tf.variable_scope("shape") as scope:
                self.z1, self.hz1 = self.shapeNetworkHolistic(self.input_shape_fea_1)
                scope.reuse_variables()
                self.z2, self.hz2 = self.shapeNetworkHolistic(self.input_shape_fea_2)

            # Create loss
            self.loss,      self.loss_1,        self.loss_2,        self.loss_3         = self.alphaDiscuss()

        elif net_type == 'hidOnly': ## only use deep metric
            with tf.variable_scope("sketch") as scope:
                self.o1, self.ho1 = self.sketchNetworkHolistic(self.input_sketch_fea_1)
                scope.reuse_variables()
                self.o2, self.ho2 = self.sketchNetworkHolistic(self.input_sketch_fea_2)
            with tf.variable_scope("shape") as scope:
                self.z1, self.hz1 = self.shapeNetworkHolistic(self.input_shape_fea_1)
                scope.reuse_variables()
                self.z2, self.hz2 = self.shapeNetworkHolistic(self.input_shape_fea_2)

            # Create loss
            self.loss,      self.loss_1,        self.loss_2,        self.loss_3         = self.hidOnly()

        elif net_type == 'metricAuto': ## use metric with auto encoder
            with tf.variable_scope("siamese1") as scope:
                self.o1, self.rx1 = self.autoMetric1(self.x1)
                scope.reuse_variables()
                self.o2, self.rx2 = self.autoMetric1(self.x2)

            with tf.variable_scope("siamese2") as scope:
                self.z1, self.ry1 = self.autoMetric2(self.y1)
                scope.reuse_variables()
                self.z2, self.ry2 = self.autoMetric2(self.y2)

        # Create loss
            self.simLabel = tf.placeholder(tf.float32, [None, 3])
            self.loss, self.loss_1, self.loss_2, self.recon_error = self.loss_with_autoencode()


    def genSimLabel(self, label1, label2):

        print(label1.get_shape())
        print(label2.get_shape())
        return tf.cast(tf.equal(label1, label2), tf.float32)

    def sketchNetwork(self, x):          #### for sketch network
        """
        This is the network for sketch
        """
        stddev = 0.1        # The standard deviation for initializing weights
        fc1 = self.fc_layer(x, 2000, "fc1", 0.1)
        ac1 = tf.nn.sigmoid(fc1)
        fc2 = self.fc_layer(ac1, 1000, "fc2", 0.1)
        ac2 = tf.nn.sigmoid(fc2)
        fc3 = self.fc_layer(ac2, 100, "fc3", 0.1)
        ac3 = tf.nn.sigmoid(fc3)

        return ac3

    def shapeNetwork(self, x):          ### for shape network
        """
        This is the network for shape
        """
        stddev = 0.1        # The standard deviation for initializing weights
        fc1 = self.fc_layer(x, 2000, "fc1", 0.1)
        ac1 = tf.nn.sigmoid(fc1)
        fc2 = self.fc_layer(ac1, 1000, "fc2", 0.1)
        ac2 = tf.nn.sigmoid(fc2)
        fc3 = self.fc_layer(ac2, 500, "fc3", 0.1)
        ac3 = tf.nn.sigmoid(fc3)
        fc4 = self.fc_layer(ac3, 100, "fc4", 0.1)
        ac4 = tf.nn.sigmoid(fc4)

        return ac4

    def sketchNetworkHolistic(self, x):          #### for sketch network
        stddev = 0.1
        fc1 = self.fc_layer(x, 2000, "fc1", stddev)
        ac1 = tf.nn.sigmoid(fc1)
        fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
        ac2 = tf.nn.sigmoid(fc2)
        fc3 = self.fc_layer(ac2, 100, "fc3", stddev)
        ac3 = tf.nn.sigmoid(fc3)

        return ac3, ac2

    def autoMetric1(self, x):          #### for sketch network
        stddev = 0.1
        fc1 = self.fc_layer(x, 2000, "fc1", stddev)
        ac1 = tf.nn.sigmoid(fc1)
        fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
        ac2 = tf.nn.sigmoid(fc2)
        fc3 = self.fc_layer(ac2, 100, "fc3", stddev)
        ac3 = tf.nn.sigmoid(fc3)
        dfc1 = self.fc_layer(ac2, 2000, "dfc1", stddev)
        dac1 = tf.nn.sigmoid(dfc1)
        dfc0 = self.fc_layer(dac1, 4096, 'dfc0', stddev)
        dac0 = tf.nn.tanh(dfc0)
        #return ac3, dac0

        return ac3, dfc0

    def autoMetric2(self, x):          ### for shape network
        stddev = 0.1
        fc1 = self.fc_layer(x, 2000, "fc1", stddev)
        ac1 = tf.nn.sigmoid(fc1)
        fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
        ac2 = tf.nn.sigmoid(fc2)
        fc3 = self.fc_layer(ac2, 500, "fc3", stddev)
        ac3 = tf.nn.sigmoid(fc3)
        fc4 = self.fc_layer(ac3, 100, "fc4", stddev)
        ac4 = tf.nn.sigmoid(fc4)
        dfc2 = self.fc_layer(ac3, 1000, 'dfc2', stddev)
        dac2 = tf.nn.sigmoid(dfc2)
        dfc1 = self.fc_layer(dac2, 2000, 'dfc1', stddev)
        dac1 = tf.nn.sigmoid(dfc1)
        dfc0 = self.fc_layer(dac1, 4096, 'dac1', stddev)
        dac0 = tf.nn.tanh(dfc0)

        #return ac4, dac0
        return ac4, dfc0


    def shapeNetworkHolistic(self, x):          ### for shape network
        stddev = 0.1
        fc1 = self.fc_layer(x, 2000, "fc1", stddev)
        ac1 = tf.nn.sigmoid(fc1)
        fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
        ac2 = tf.nn.sigmoid(fc2)
        fc3 = self.fc_layer(ac2, 500, "fc3", stddev)
        ac3 = tf.nn.sigmoid(fc3)
        fc4 = self.fc_layer(ac3, 100, "fc4", stddev)
        ac4 = tf.nn.sigmoid(fc4)

        return ac4, ac2

    def fc_layer(self, bottom, n_weight, name, stddev):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        #initer = tf.truncated_normal_initializer(stddev=0.01)
        initer = tf.truncated_normal_initializer(stddev=stddev)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        ## rescale weight by 0.1
    #    W = tf.mul(.1, W)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.0, shape=[n_weight], dtype=tf.float32))
        ## rescale biase by 0.1
    #    b = tf.mul(.1, b)
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)

        return fc

    def contrastiveLoss(self):

        def constructLoss(fea1, fea2, simLabel, margin, lossName):
            disLabel = tf.subtract(1.0, simLabel)
            eucDist = tf.expand_dims(tf.reduce_sum(tf.square(tf.subtract(fea1, fea2)), axis=1), axis=-1)

            pos_dist = tf.multiply(simLabel, eucDist)
            neg_dist = tf.multiply(disLabel, tf.maximum(tf.subtract(margin, eucDist), 0.))
            loss = tf.reduce_mean(tf.add(pos_dist, neg_dist), name=lossName)
            return pos_dist, neg_dist, loss


        # loss function for sketch
        pos_sketch, neg_sketch, loss_sketch = constructLoss(self.o1, self.o2, self.sketch_simLabel, self.margin_out, 'loss_sketch')

        # loss function for shape
        pos_shape, neg_shape, loss_shape = constructLoss(self.z1, self.z2, self.shape_simLabel, self.margin_out, 'loss_shape')

        # loss function cross domain

        pos_cross_1, neg_cross_1, loss_cross_1 = constructLoss(self.o1, self.z1, self.sketch_shape_simLabel_1, self.margin_out, 'loss_cross_1')

        pos_cross_2, neg_cross_2, loss_cross_2 = constructLoss(self.o2, self.z2, self.sketch_shape_simLabel_2, self.margin_out, 'loss_cross_2')

        # for cross    #############
        loss = tf.add_n([loss_sketch, loss_shape, loss_cross_1, loss_cross_2], name='loss')

        return loss, loss_sketch, loss_shape, loss_cross_1

    def hidOnly(self):


        def constructLoss(fea1, fea2, simLabel, margin, lossName):

            disLabel = tf.subtract(1.0, simLabel)
            eucDist = tf.expand_dims(tf.reduce_sum(tf.square(tf.subtract(fea1, fea2)), axis=1), axis=-1)

            pos_dist = tf.multiply(simLabel, eucDist)
            neg_dist = tf.multiply(disLabel, tf.maximum(tf.subtract(margin, eucDist), 0.))
            loss = tf.reduce_mean(tf.add(pos_dist, neg_dist), name=lossName)

            return pos_dist, neg_dist, loss


        # For output layer

        # loss function for sketch
        pos_sketch_out, neg_sketch_out, loss_sketch_out = constructLoss(self.o1, self.o2, self.sketch_simLabel, self.margin_out, 'loss_sketch_out')

        # loss function for shape
        pos_shape_out, neg_shape_out, loss_shape_out = constructLoss(self.z1, self.z2, self.shape_simLabel, self.margin_out, 'loss_shape_out')

        # loss function cross domain

        # pos_cross_out_1, neg_cross_out_1, loss_cross_out_1 = constructLoss(self.o1, self.z1, self.sketch_shape_simLabel_1, self.margin_out, 'loss_cross_out_1')

        # pos_cross_out_2, neg_cross_out_2, loss_cross_out_2 = constructLoss(self.o2, self.z2, self.sketch_shape_simLabel_2, self.margin_out, 'loss_cross_out_2')

        # for cross    #############
        # loss_out = tf.add_n([loss_sketch_out, loss_shape_out, loss_cross_out_1, loss_cross_out_2], name='loss_out')

        loss_out = tf.add_n([loss_sketch_out, loss_shape_out], name='loss_out')


        # For hidden layer

        # loss function for sketch
        pos_sketch_hid, neg_sketch_hid, loss_sketch_hid = constructLoss(self.ho1, self.ho2, self.sketch_simLabel, self.margin_hid, 'loss_sketch_hid')

        # loss function for shape
        pos_shape_hid, neg_shape_hid, loss_shape_hid = constructLoss(self.hz1, self.hz2, self.shape_simLabel, self.margin_hid, 'loss_shape_hid')

        # loss function cross domain

        pos_cross_hid_1, neg_cross_hid_1, loss_cross_hid_1 = constructLoss(self.ho1, self.hz1, self.sketch_shape_simLabel_1, self.margin_hid, 'loss_cross_hid_1')

        pos_cross_hid_2, neg_cross_hid_2, loss_cross_hid_2 = constructLoss(self.ho2, self.hz2, self.sketch_shape_simLabel_2, self.margin_hid, 'loss_cross_hid_2')

        # for cross    #############
        loss_hid = tf.add_n([loss_sketch_hid, loss_shape_hid, loss_cross_hid_1, loss_cross_hid_2], name='loss_hid')



        loss = tf.add(loss_out, loss_hid, name='loss')
        loss_sketch = tf.add(loss_sketch_out, loss_sketch_hid, name='loss_sketch')
        loss_shape = tf.add(loss_shape_out, loss_shape_hid, name='loss_shape')
        loss_cross_1 = loss_cross_hid_1
        loss_cross_2 = loss_cross_hid_2

        return loss, loss_sketch, loss_shape, loss_cross_1

    def alphaDiscuss(self):


        def constructLoss(fea1, fea2, simLabel, margin, lossName):

            disLabel = tf.subtract(1.0, simLabel)
            eucDist = tf.expand_dims(tf.reduce_sum(tf.square(tf.subtract(fea1, fea2)), axis=1), axis=-1)

            pos_dist = tf.multiply(simLabel, eucDist)
            neg_dist = tf.multiply(disLabel, tf.maximum(tf.subtract(margin, eucDist), 0.))
            loss = tf.reduce_mean(tf.add(pos_dist, neg_dist), name=lossName)

            return pos_dist, neg_dist, loss


        # For output layer

        # loss function for sketch
        pos_sketch_out, neg_sketch_out, loss_sketch_out = constructLoss(self.o1, self.o2, self.sketch_simLabel, self.margin_out, 'loss_sketch_out')

        # loss function for shape
        pos_shape_out, neg_shape_out, loss_shape_out = constructLoss(self.z1, self.z2, self.shape_simLabel, self.margin_out, 'loss_shape_out')

        # loss function cross domain

        pos_cross_out_1, neg_cross_out_1, loss_cross_out_1 = constructLoss(self.o1, self.z1, self.sketch_shape_simLabel_1, self.margin_out, 'loss_cross_out_1')

        pos_cross_out_2, neg_cross_out_2, loss_cross_out_2 = constructLoss(self.o2, self.z2, self.sketch_shape_simLabel_2, self.margin_out, 'loss_cross_out_2')

        # for cross    #############
        loss_out = tf.add_n([loss_sketch_out, loss_shape_out, loss_cross_out_1, loss_cross_out_2], name='loss_out')


        # For hidden layer

        # loss function for sketch
        pos_sketch_hid, neg_sketch_hid, loss_sketch_hid = constructLoss(self.ho1, self.ho2, self.sketch_simLabel, self.margin_hid, 'loss_sketch_hid')

        # loss function for shape
        pos_shape_hid, neg_shape_hid, loss_shape_hid = constructLoss(self.hz1, self.hz2, self.shape_simLabel, self.margin_hid, 'loss_shape_hid')

        # loss function cross domain

        pos_cross_hid_1, neg_cross_hid_1, loss_cross_hid_1 = constructLoss(self.ho1, self.hz1, self.sketch_shape_simLabel_1, self.margin_hid, 'loss_cross_hid_1')

        pos_cross_hid_2, neg_cross_hid_2, loss_cross_hid_2 = constructLoss(self.ho2, self.hz2, self.sketch_shape_simLabel_2, self.margin_hid, 'loss_cross_hid_2')

        # for cross    #############
        loss_hid = tf.add_n([loss_sketch_hid, loss_shape_hid, loss_cross_hid_1, loss_cross_hid_2], name='loss_hid')



        loss = tf.add(loss_out, loss_hid, name='loss')
        loss_sketch = tf.add(loss_sketch_out, loss_sketch_hid, name='loss_sketch')
        loss_shape = tf.add(loss_shape_out, loss_shape_hid, name='loss_shape')
        loss_cross_1 = tf.add(tf.multiply(self.alpha, loss_cross_out_1), tf.multiply(1.-self.alpha, loss_cross_hid_1), name='loss_cross_1')
        loss_cross_2 = tf.add(tf.multiply(self.alpha, loss_cross_out_2), tf.multiply(1.-self.alpha, loss_cross_hid_2), name='loss_cross_2')

        return loss, loss_sketch, loss_shape, loss_cross_1

    def contrastiveLossHolistic(self):


        def constructLoss(fea1, fea2, simLabel, margin, lossName):

            disLabel = tf.subtract(1.0, simLabel)
            eucDist = tf.expand_dims(tf.reduce_sum(tf.square(tf.subtract(fea1, fea2)), axis=1), axis=-1)

            pos_dist = tf.multiply(simLabel, eucDist)
            neg_dist = tf.multiply(disLabel, tf.maximum(tf.subtract(margin, eucDist), 0.))
            loss = tf.reduce_mean(tf.add(pos_dist, neg_dist), name=lossName)

            return pos_dist, neg_dist, loss


        # For output layer

        # loss function for sketch
        pos_sketch_out, neg_sketch_out, loss_sketch_out = constructLoss(self.o1, self.o2, self.sketch_simLabel, self.margin_out, 'loss_sketch_out')

        # loss function for shape
        pos_shape_out, neg_shape_out, loss_shape_out = constructLoss(self.z1, self.z2, self.shape_simLabel, self.margin_out, 'loss_shape_out')

        # loss function cross domain

        pos_cross_out_1, neg_cross_out_1, loss_cross_out_1 = constructLoss(self.o1, self.z1, self.sketch_shape_simLabel_1, self.margin_out, 'loss_cross_out_1')

        pos_cross_out_2, neg_cross_out_2, loss_cross_out_2 = constructLoss(self.o2, self.z2, self.sketch_shape_simLabel_2, self.margin_out, 'loss_cross_out_2')

        # for cross    #############
        loss_out = tf.add_n([loss_sketch_out, loss_shape_out, loss_cross_out_1, loss_cross_out_2], name='loss_out')


        # For hidden layer

        # loss function for sketch
        pos_sketch_hid, neg_sketch_hid, loss_sketch_hid = constructLoss(self.ho1, self.ho2, self.sketch_simLabel, self.margin_hid, 'loss_sketch_hid')

        # loss function for shape
        pos_shape_hid, neg_shape_hid, loss_shape_hid = constructLoss(self.hz1, self.hz2, self.shape_simLabel, self.margin_hid, 'loss_shape_hid')

        # loss function cross domain

        pos_cross_hid_1, neg_cross_hid_1, loss_cross_hid_1 = constructLoss(self.ho1, self.hz1, self.sketch_shape_simLabel_1, self.margin_hid, 'loss_cross_hid_1')

        pos_cross_hid_2, neg_cross_hid_2, loss_cross_hid_2 = constructLoss(self.ho2, self.hz2, self.sketch_shape_simLabel_2, self.margin_hid, 'loss_cross_hid_2')

        # for cross    #############
        loss_hid = tf.add_n([loss_sketch_hid, loss_shape_hid, loss_cross_hid_1, loss_cross_hid_2], name='loss_hid')



        loss = tf.add(loss_out, loss_hid, name='loss')
        loss_sketch = tf.add(loss_sketch_out, loss_sketch_hid, name='loss_sketch')
        loss_shape = tf.add(loss_shape_out, loss_shape_hid, name='loss_shape')
        loss_cross_1 = tf.add(loss_cross_out_1, loss_cross_hid_1, name='loss_cross_1')
        loss_cross_2 = tf.add(loss_cross_out_2, loss_cross_hid_2, name='loss_cross_2')

        return loss, loss_sketch, loss_shape, loss_cross_1

    def holisticLoss(self):
        #### for the siamese network 1      #############
        # margin = 5.0
        margin = 3.0
        margin_h = 50.0
        #margin = 1.0
	#margin = 2.0

        # For output layer ##
        labels_t = self.simLabel[:,0]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_1 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        C = tf.constant(margin)
        #eucd2_1 = tf.reduce_sum(eucd2_1, 1)
        eucd2_1 = tf.reduce_sum(eucd2_1, 1)
        pos_1 = tf.multiply(labels_t, eucd2_1)
        ### assign different weight for positive term
        #pos_1 = tf.mul(10.0, pos_1)
        #neg_1 = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_1), 0), 2))
        neg_1 = tf.multiply(labels_f, tf.maximum(tf.subtract(C, eucd2_1), 0))
        ### assign different weight for negative term
        #neg_1 = tf.mul(0.01, neg_1)

        losses_1 = tf.add(pos_1, neg_1, name="losses_1")
        loss_1 = tf.reduce_mean(losses_1)

        # For hidden layer ##
        labels_t = self.simLabel[:,0]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_1 = tf.pow(tf.subtract(self.ho1, self.ho2), 2)
        C = tf.constant(margin_h)
        #eucd2_1 = tf.reduce_sum(eucd2_1, 1)
        eucd2_1 = tf.reduce_sum(eucd2_1, 1)
        pos_1 = tf.multiply(labels_t, eucd2_1)
        ### assign different weight for positive term
        #pos_1 = tf.mul(10.0, pos_1)
        #neg_1 = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_1), 0), 2))
        neg_1 = tf.multiply(labels_f, tf.maximum(tf.subtract(C, eucd2_1), 0))
        ### assign different weight for negative term
        #neg_1 = tf.mul(0.01, neg_1)

        losses_1 = tf.add(pos_1, neg_1, name="losses_1")
        loss_h1 = tf.reduce_mean(losses_1)



        #### for the siamese network 2      #############
        # For output layer
        C_shape = tf.constant(margin)
        labels_t = self.simLabel[:,1]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_2 = tf.pow(tf.subtract(self.z1, self.z2), 2)
        eucd2_2 = tf.reduce_sum(eucd2_2, 1)
        pos_2 = tf.multiply(labels_t, eucd2_2)
        #neg_2 = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_2), 0), 2))
        neg_2 = tf.multiply(labels_f, tf.maximum(tf.subtract(C_shape, eucd2_2), 0))

        losses_2 = tf.add(pos_2, neg_2, name="losses_2")
        loss_2 = tf.reduce_mean(losses_2)
        # For hidden layer
        C_shape = tf.constant(margin_h)
        labels_t = self.simLabel[:,1]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_2 = tf.pow(tf.subtract(self.hz1, self.hz2), 2)
        eucd2_2 = tf.reduce_sum(eucd2_2, 1)
        pos_2 = tf.multiply(labels_t, eucd2_2)
        #neg_2 = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_2), 0), 2))
        neg_2 = tf.multiply(labels_f, tf.maximum(tf.subtract(C_shape, eucd2_2), 0))

        losses_2 = tf.add(pos_2, neg_2, name="losses_2")
        loss_h2 = tf.reduce_mean(losses_2)


        #### for the siamese network 12      #############
        # For output layer
        labels_t = self.simLabel[:,2]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_3 = tf.pow(tf.subtract(self.o1, self.z1), 2)
        eucd2_3 = tf.reduce_sum(eucd2_3, 1)
        pos_3 = tf.multiply(labels_t, eucd2_3)
        #neg_3 = tf.multiply(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_3), 0), 2))
        neg_3 = tf.multiply(labels_f, tf.maximum(tf.subtract(margin, eucd2_3), 0))
        losses_3 = tf.add(pos_3, neg_3, name="losses_3")
        loss_3 = tf.reduce_mean(losses_3)
        # For hidden layer
        labels_t = self.simLabel[:,2]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_3 = tf.pow(tf.subtract(self.ho1, self.hz1), 2)
        eucd2_3 = tf.reduce_sum(eucd2_3, 1)
        pos_3 = tf.multiply(labels_t, eucd2_3)
        #neg_3 = tf.multiply(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_3), 0), 2))
        neg_3 = tf.multiply(labels_f, tf.maximum(tf.subtract(margin_h, eucd2_3), 0))
        losses_3 = tf.add(pos_3, neg_3, name="losses_3")
        loss_h3 = tf.reduce_mean(losses_3)


        #############################################################################
        loss_all_1 = tf.add(loss_1, loss_h1)
        loss_all_2 = tf.add(loss_2, loss_h2)
        loss_all_3 = tf.add(loss_3, loss_h3)

        loss = tf.add(tf.add(loss_all_1, loss_all_2), loss_all_3, name="losses")

        return loss, loss_all_1, loss_all_2

    def loss_with_autoencode(self):
        #### for the siamese network 1      #############
        margin = 3.0
        labels_t = self.simLabel[:,0]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_1 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        C = tf.constant(margin)
        #eucd2_1 = tf.reduce_sum(eucd2_1, 1)
        eucd2_1 = tf.reduce_sum(eucd2_1, 1)
        pos_1 = tf.multiply(labels_t, eucd2_1)
        ### assign different weight for positive term
        #pos_1 = tf.multiply(10.0, pos_1)
        #neg_1 = tf.multiply(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_1), 0), 2))
        neg_1 = tf.multiply(labels_f, tf.maximum(tf.subtract(C, eucd2_1), 0))
        ### assign different weight for negative term
        #neg_1 = tf.multiply(0.01, neg_1)

        losses_1 = tf.add(pos_1, neg_1, name="losses_1")
        loss_1 = tf.reduce_mean(losses_1)
        #### for the siamese network 2      #############

        C_shape = tf.constant(margin)
        labels_t = self.simLabel[:,1]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_2 = tf.pow(tf.subtract(self.z1, self.z2), 2)
        eucd2_2 = tf.reduce_sum(eucd2_2, 1)
        pos_2 = tf.multiply(labels_t, eucd2_2)
        #neg_2 = tf.multiply(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_2), 0), 2))
        neg_2 = tf.multiply(labels_f, tf.maximum(tf.subtract(C_shape, eucd2_2), 0))

        losses_2 = tf.add(pos_2, neg_2, name="losses_2")
        loss_2 = tf.reduce_mean(losses_2)

        #### for the siamese network 12      #############
        labels_t = self.simLabel[:,2]
        labels_f = tf.subtract(1.0, labels_t)          # labels_ = !labels;
        eucd2_3 = tf.pow(tf.subtract(self.o1, self.z1), 2)
        eucd2_3 = tf.reduce_sum(eucd2_3, 1)
        pos_3 = tf.multiply(labels_t, eucd2_3)
        #neg_3 = tf.multiply(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd2_3), 0), 2))
        neg_3 = tf.multiply(labels_f, tf.maximum(tf.subtract(C, eucd2_3), 0))

        losses_3 = tf.add(pos_3, neg_3, name="losses_3")
        #############################################################################
        losses = tf.add(tf.add(losses_1, losses_2), losses_3, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        ############# For autoencoder reconstruction loss ##########################

        diff_sketch = tf.pow(tf.subtract(self.x1, self.rx1), 2) + tf.pow(tf.subtract(self.x2, self.rx2), 2)
        diff_sketch = tf.reduce_sum(diff_sketch, 1)
        error_sketch = tf.reduce_mean(diff_sketch)
        #### shape ##########
        diff_shape = tf.pow(tf.subtract(self.y1, self.ry1), 2) + tf.pow(tf.subtract(self.y2, self.ry2), 2)
        diff_shape = tf.reduce_sum(diff_shape, 1)
        error_shape = tf.reduce_mean(diff_shape)
        ############################################
        recon_error = error_sketch + error_shape
        loss = tf.add(loss, 1e-3*recon_error)

        return loss, loss_1, loss_2, recon_error
