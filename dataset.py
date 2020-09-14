import numpy as np
# import cv2
from random import shuffle
import scipy.io as sio
import time
import sys
import math

class Dataset:

    def __init__(self, sketch_train_list, sketch_test_list, shape_list, inputFeaSize, class_num):

        self.inputFeaSize = inputFeaSize
        self.class_num    = class_num

        # Load training images (path) and labels
        with open(sketch_train_list) as f:
            lines = f.readlines()
            ### only pick 5 groups for training####
#            lines = lines[:250]
        self.sketch_train_fea = []
        self.sketch_train_label = []
        shuffle(lines)
        self.sketch_train_data = lines
        self.sketch_train_size = len(lines)
        self.sketch_train_randIndex = np.random.permutation(self.sketch_train_size)
        for l in lines:
            items = l.split()
            self.sketch_train_fea.append(items[0])
            self.sketch_train_label.append(items[1])
        ### Load test sketch ##############3
        with open(sketch_test_list) as f:
            lines = f.readlines()
            ### only pick 5 groups for training####
#            lines = lines[:150]
        self.sketch_test_fea = []
        self.sketch_test_label = []
        shuffle(lines)
        self.sketch_test_data = lines
        self.sketch_test_size = len(lines)
        self.sketch_test_randIndex = np.random.permutation(self.sketch_test_size)
        for l in lines:
            items = l.split()
            self.sketch_test_fea.append(items[0])
            self.sketch_test_label.append(items[1])
        with open(shape_list) as f:
            lines = f.readlines()
            #### only pick g groups ####
#            lines = lines[:202]
        self.shape_fea = []
        self.shape_label = []
        shuffle(lines)
        self.shape_data = lines
        self.shape_size = len(lines)
        self.shape_randIndex = np.random.permutation(self.shape_size)
        for l in lines:
            items = l.split()
            self.shape_fea.append(items[0])
            self.shape_label.append(items[1])
        self.sketch_train_ptr = 0       ## pointer training sketch
        self.sketch_test_ptr = 0       ## pointer for testing sketch
        self.shape_ptr = 0
        ### whether shuffle data at the start of each epoch


        self.loadAllData()
        self.shuffle = 1



    def getGroupByClass(self):
        self.shape_dict = {}
        self.sketch_train_dict = {}
        for i in range(self.class_num):
            self.shape_dict[str(i)] = {}            # empty dictionary for each class
            self.shape_dict[str(i)]['index'] = np.where(self.shapeLabelset == i)
            self.shape_dict[str(i)]['sampleNum'] = np.where(self.shapeLabelset == i)[0].size
            self.sketch_train_dict[str(i)] = {}
            self.sketch_train_dict[str(i)]['index'] = np.where(self.sketchTrainLabelset == i)
            self.sketch_train_dict[str(i)]['sampleNum'] = np.where(self.sketchTrainLabelset == i)[0].size



    def next_batch_cor(self, batch_size):

        # Get next batch of image (path) and labels
        shapeIndex1 = np.array([])
        shapeIndex2 = np.array([])

        sketchIndex1 = np.array([])
        sketchIndex2 = np.array([])

        for i in range(self.class_num):
            # The total number of examples for each class
            sampleNum = self.shape_dict[str(i)]['sampleNum']
            shapeRandIndex1 = np.random.permutation(math.max(sampleNum, batch_size))[:math.min(sampleNum, batch_size)]
            shapeIndex1 = np.concatenate([shapeIndex1, shapeRandIndex1])

            shapeRandIndex2 = np.random.permutation(math.max(sampleNum, batch_size))[:math.min(sampleNum, batch_size)]
            shapeIndex2 = np.concatenate([shapeIndex2, shapeRandIndex2])

            sampleNum = self.sketch_train_dict[str(i)]['sampleNum']
            sketchRandIndex1 = np.random.permutation(math.max(sampleNum, batch_size))[:math.min(sampleNum, batch_size)]
            sketchIndex1 = np.concatenate([sketchIndex1, sketchRandIndex1])
            sketchRandIndex2 = np.random.permutation(math.max(sampleNum, batch_size))[:math.min(sampleNum, batch_size)]
            sketchIndex2 = np.concatenate([sketchIndex2, sketchRandIndex2])



        # To be changed 

        for i in range(self.class_num):
            # The total number of examples for each class
            sampleNum = self.shape_dict[str(i)]['sampleNum']
            shapeRandIndex1 = (np.random.permutation(math.max(sampleNum, batch_size)) % math.min(sampleNum, batch_size))[:batch_size]
            shapeIndex1 = np.concatenate([shapeIndex1, shapeRandIndex1])

            shapeRandIndex2 = (np.random.permutation(math.max(sampleNum, batch_size)) % math.min(sampleNum, batch_size))[:batch_size]
            shapeIndex2 = np.concatenate([shapeIndex2, shapeRandIndex2])

            sampleNum = self.sketch_train_dict[str(i)]['sampleNum']
            sketchRandIndex1 = (np.random.permutation(math.max(sampleNum, batch_size))%math.min(sampleNum, batch_size))[:batch_size]
            sketchIndex1 = np.concatenate([sketchIndex1, sketchRandIndex1])
            sketchRandIndex2 = (np.random.permutation(math.max(sampleNum, batch_size))%math.min(sampleNum, batch_size))[:batch_size]
            sketchIndex2 = np.concatenate([sketchIndex2, sketchRandIndex2])













        def getBatch(start_pointer, batch_size, dataset_size, randomIndexAll, feaset, labelset):
            if start_pointer + batch_size < dataset_size:
                randIndex = randomIndexAll[start_pointer:start_pointer + batch_size]
                images = feaset[randIndex]
                labels = labelset[randIndex]
                start_pointer += batch_size

                return images, labels, start_pointer
            else:
                new_ptr = (start_pointer + batch_size)%dataset_size
                randIndex = np.concatenate((randomIndexAll[start_pointer:], randomIndexAll[:new_ptr]), axis=0)
                images = feaset[randIndex]
                labels = labelset[randIndex]
                start_pointer = new_ptr
                randomIndexAll = np.random.permutation(dataset_size)

                return images, labels, start_pointer, randomIndexAll

        if phase == 'sketch_train':
            if self.sketch_train_ptr + batch_size < self.sketch_train_size:
                images, labels, self.sketch_train_ptr = getBatch(self.sketch_train_ptr, batch_size,
                        self.sketch_train_size, self.sketch_train_randIndex,
                        self.sketchTrainFeaset, self.sketchTrainLabelset)
            else:

                images, labels, self.sketch_train_ptr, self.sketch_train_randIndex = getBatch(self.sketch_train_ptr, batch_size,
                        self.sketch_train_size, self.sketch_train_randIndex,
                        self.sketchTrainFeaset, self.sketchTrainLabelset)
        elif phase == 'sketch_test':
            if self.sketch_test_ptr + batch_size < self.sketch_test_size:
                images, labels, self.sketch_test_ptr = getBatch(self.sketch_test_ptr, batch_size,
                        self.sketch_test_size, self.sketch_test_randIndex,
                        self.sketchTestFeaset, self.sketchTestLabelset)
            else:
                images, labels, self.sketch_test_ptr, self.sketch_test_randIndex = getBatch(self.sketch_test_ptr, batch_size,
                        self.sketch_test_size, self.sketch_test_randIndex,
                        self.sketchTestFeaset, self.sketchTestLabelset)

        elif phase == 'shape':
            if self.shape_ptr + batch_size < self.shape_size:
                images, labels, self.shape_ptr = getBatch(self.shape_ptr, batch_size,
                        self.shape_size, self.shape_randIndex,
                        self.shapeFeaset, self.shapeLabelset)
            else:
                images, labels, self.shape_ptr, self.shape_randIndex = getBatch(self.shape_ptr, batch_size,
                        self.shape_size, self.shape_randIndex,
                        self.shapeFeaset, self.shapeLabelset)


        return images, labels



    def loadAllData(self):
        def loadFeaAndLabel(pathSet, feaSize, phase):
            def loaddata(filepath, phase):
                mat_contents = sio.loadmat(filepath)
            # print(mat_contents)
                if phase == 'sketch_train' or phase == 'sketch_test':
                    img = mat_contents['fea']
                elif phase == 'shape':
                    img = mat_contents['coeff']

                return img.flatten()


            sampleNum = len(pathSet)
            feaSet = np.zeros((sampleNum, feaSize))
            labelSet = np.zeros((sampleNum, 1))
            for i in range(sampleNum):
                filePath = pathSet[i].split(' ')
                feaSet[i] = loaddata(filePath[0], phase)
                labelSet[i] = int(filePath[1])
            return feaSet, labelSet
        print("Load sketch testing features")
        start_time = time.time()
        print(self.inputFeaSize)
        self.sketchTestFeaset, self.sketchTestLabelset = loadFeaAndLabel(self.sketch_test_data, self.inputFeaSize, 'sketch_test')
        print("Loading time: {}".format(time.time() - start_time))

        print("Load sketch training features")
        start_time = time.time()
        self.sketchTrainFeaset, self.sketchTrainLabelset = loadFeaAndLabel(self.sketch_train_data, self.inputFeaSize, 'sketch_train')
        print("Loading time: {}".format(time.time() - start_time))

        print("Load shape features")
        start_time = time.time()
        self.shapeFeaset, self.shapeLabelset = loadFeaAndLabel(self.shape_data, self.inputFeaSize, 'shape')
        print("Loading time: {}".format(time.time() - start_time))

        print("Finish Loading")


    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels

        def getBatch(start_pointer, batch_size, dataset_size, randomIndexAll, feaset, labelset):
            if start_pointer + batch_size < dataset_size:
                randIndex = randomIndexAll[start_pointer:start_pointer + batch_size]
                images = feaset[randIndex]
                labels = labelset[randIndex]
                start_pointer += batch_size

                return images, labels, start_pointer
            else:
                new_ptr = (start_pointer + batch_size)%dataset_size
                randIndex = np.concatenate((randomIndexAll[start_pointer:], randomIndexAll[:new_ptr]), axis=0)
                images = feaset[randIndex]
                labels = labelset[randIndex]
                start_pointer = new_ptr
                randomIndexAll = np.random.permutation(dataset_size)

                return images, labels, start_pointer, randomIndexAll

        if phase == 'sketch_train':
            if self.sketch_train_ptr + batch_size < self.sketch_train_size:
                images, labels, self.sketch_train_ptr = getBatch(self.sketch_train_ptr, batch_size,
                        self.sketch_train_size, self.sketch_train_randIndex,
                        self.sketchTrainFeaset, self.sketchTrainLabelset)
            else:

                images, labels, self.sketch_train_ptr, self.sketch_train_randIndex = getBatch(self.sketch_train_ptr, batch_size,
                        self.sketch_train_size, self.sketch_train_randIndex,
                        self.sketchTrainFeaset, self.sketchTrainLabelset)
        elif phase == 'sketch_test':
            if self.sketch_test_ptr + batch_size < self.sketch_test_size:
                images, labels, self.sketch_test_ptr = getBatch(self.sketch_test_ptr, batch_size,
                        self.sketch_test_size, self.sketch_test_randIndex,
                        self.sketchTestFeaset, self.sketchTestLabelset)
            else:
                images, labels, self.sketch_test_ptr, self.sketch_test_randIndex = getBatch(self.sketch_test_ptr, batch_size,
                        self.sketch_test_size, self.sketch_test_randIndex,
                        self.sketchTestFeaset, self.sketchTestLabelset)

        elif phase == 'shape':
            if self.shape_ptr + batch_size < self.shape_size:
                images, labels, self.shape_ptr = getBatch(self.shape_ptr, batch_size,
                        self.shape_size, self.shape_randIndex,
                        self.shapeFeaset, self.shapeLabelset)
            else:
                images, labels, self.shape_ptr, self.shape_randIndex = getBatch(self.shape_ptr, batch_size,
                        self.shape_size, self.shape_randIndex,
                        self.shapeFeaset, self.shapeLabelset)


        return images, labels

    def next_batch_backup(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'sketch_train':
            if self.sketch_train_ptr + batch_size < self.sketch_train_size:
                paths = self.sketch_train_fea[self.sketch_train_ptr:self.sketch_train_ptr + batch_size]
                labels = self.sketch_train_label[self.sketch_train_ptr:self.sketch_train_ptr + batch_size]
                self.sketch_train_ptr += batch_size
            else:
                new_ptr = (self.sketch_train_ptr + batch_size)%self.sketch_train_size
                paths = self.sketch_train_fea[self.sketch_train_ptr:] + self.sketch_train_fea[:new_ptr]
                labels = self.sketch_train_label[self.sketch_train_ptr:] + self.sketch_train_label[:new_ptr]
                self.sketch_train_ptr = new_ptr
                if self.shuffle == 1:
                    c = list(zip(self.sketch_train_fea, self.sketch_train_label))
                    shuffle(c)
                    self.sketch_train_fea, self.sketch_train_label = zip(*c)
        elif phase == 'sketch_test':
            if self.sketch_test_ptr + batch_size < self.sketch_test_size:
                paths = self.sketch_test_fea[self.sketch_test_ptr:self.sketch_test_ptr + batch_size]
                labels = self.sketch_test_label[self.sketch_test_ptr:self.sketch_test_ptr + batch_size]
                self.sketch_test_ptr += batch_size
            else:
                new_ptr = (self.sketch_test_ptr + batch_size)%self.sketch_test_size
                paths = self.sketch_test_fea[self.sketch_test_ptr:] + self.sketch_test_fea[:new_ptr]
                labels = self.sketch_test_label[self.sketch_test_ptr:] + self.sketch_test_label[:new_ptr]
                self.sketch_test_ptr = new_ptr
                if self.shuffle == 1:
                    c = list(zip(self.sketch_test_fea, self.sketch_test_label))
                    shuffle(c)
                    self.sketch_test_fea, self.sketch_test_label = zip(*c)

        elif phase == 'shape':
            if self.shape_ptr + batch_size < self.shape_size:
                paths = self.shape_fea[self.shape_ptr:self.shape_ptr + batch_size]
                labels = self.shape_label[self.shape_ptr:self.shape_ptr + batch_size]
                self.shape_ptr += batch_size
            else:
                new_ptr = (self.shape_ptr + batch_size)%self.shape_size
                paths = self.shape_fea[self.shape_ptr:] + self.shape_fea[:new_ptr]
                labels = self.shape_label[self.shape_ptr:] + self.shape_label[:new_ptr]
                self.shape_ptr = new_ptr
                if self.shuffle == 1:
                    c = list(zip(self.shape_fea, self.shape_label))
                    shuffle(c)
                    self.shape_fea, self.shape_label = zip(*c)

        else:
            return None, None

        # Read images
        images = np.ndarray([batch_size, 4096])     ### 4096 is the feature size
        for i in xrange(len(paths)):
            mat_contents = sio.loadmat(paths[i])
            # print(mat_contents)
            if phase == 'sketch_train' or phase == 'sketch_test':
                img = mat_contents['fea']
            elif phase == 'shape':
                img = mat_contents['coeff']
            #print('The feature shape is {}'.format(img.shape))
            h, feaLength = img.shape
            assert feaLength==4096
            images[i] = img
        labels = np.reshape(np.array(labels), (batch_size, 1))
        #print(labels)
        #print(labels.shape)
        # Expand labels
        return images, labels
    def retrievalParamSP(self):
        shapeLabels = np.array(self.shape_label)            ### cast all the labels as array
        sketchTestLabel = np.array(self.sketch_test_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):             ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0] ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]      ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths
    def retrievalParamSS(self):
        shapeLabels = np.array(self.sketch_train_label)            ### cast all the labels as array
        sketchTestLabel = np.array(self.sketch_test_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):             ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0] ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]      ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths
    def retrievalParamPP(self):
        shapeLabels = np.array(self.shape_label)            ### cast all the labels as array
        sketchTestLabel = np.array(self.shape_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):             ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0] ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]      ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths


    def normalizeData(self):
        ########### normalize sketch test feature ######################
        #print('Processing testing sketch\n')
        sketch_test_feaset = np.zeros((self.sketch_test_size, 4096))
        for i in range(len(self.sketch_test_label)):
            # print(i)
            mat_contents = sio.loadmat(self.sketch_test_fea[i])
            img = mat_contents['fea']
            #print(img)
            sketch_test_feaset[i] = img
        sketch_test_mean = np.mean(sketch_test_feaset, axis=0)
        sketch_test_std = np.std(sketch_test_feaset, axis=0)
        sketch_test_feaset_norm = (sketch_test_feaset - sketch_test_mean) / sketch_test_std
        #print(np.where(np.isnan(sketch_test_feaset_norm)))
        ########### nomralize sketch train feature #####################
        #print('Processing training sketch')
        sketch_train_feaset = np.zeros((self.sketch_train_size, 4096))
        for i in range(len(self.sketch_train_label)):
            # print(i)
            mat_contents = sio.loadmat(self.sketch_train_fea[i])
            img = mat_contents['fea']
            #print(img)
            sketch_train_feaset[i] = img
        sketch_train_mean = np.mean(sketch_train_feaset, axis=0)
        sketch_train_std = np.std(sketch_train_feaset, axis=0)
        sketch_train_feaset_norm = (sketch_train_feaset - sketch_train_mean) / sketch_train_std
        #print(np.where(np.isnan(sketch_train_feaset_norm)))
        ########## normalize shape feature ###################
        #print('Processing shape\n')
        shape_feaset = np.zeros((self.shape_size, 4096))
        for i in range(len(self.shape_label)):
            # print(i)
            mat_contents = sio.loadmat(self.shape_fea[i])
            img = mat_contents['coeff']
            shape_feaset[i] = img
        shape_mean = np.mean(shape_feaset, axis=0)
        shape_std = np.std(shape_feaset, axis=0)
        shape_feaset_norm = (shape_feaset - shape_mean) / shape_std
        ###### get rid of nan Dataset################
        shape_feaset_norm[np.where(np.isnan(shape_feaset_norm))] = 0
        #print(np.where(np.isnan(shape_feaset_norm)))
        return sketch_test_mean, sketch_test_std, sketch_train_mean, sketch_train_std, shape_mean, shape_std
