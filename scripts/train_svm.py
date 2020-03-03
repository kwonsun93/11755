#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
from sklearn.metrics.pairwise import chi2_kernel, laplacian_kernel, additive_chi2_kernel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
import _pickle as cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0]))
        print("event_name -- name of the event (P001, P002 or P003 in Homework 1)")
        print("feat_dir -- dir of feature files")
        print("feat_dim -- dim of features")
        print("output_file -- path to save the svm model")
        exit(1)

    event_name = sys.argv[1]     # P001, P002, P003
    feat_dir = sys.argv[2]       # ./kmeans/ or ./asrfeat/
    feat_dim = int(sys.argv[3])  # 200
    output_file = sys.argv[4]    # mfcc_pred/svm.$event_mfcc.lst

    # data loading
    train_list = open('/home/ubuntu/11775-hws/all_trn.lst', 'r')
    x = []
    y = []
    for file in train_list.readlines():
        filename, label = file[:-1].split(' ')
        data_path = feat_dir + filename + '.feats'
        data = np.genfromtxt(data_path, delimiter=';')
        #if sum(data)>0:
        #    data = data/sum(data)
        if label == event_name:
            x.append(data)
            y.append(1)
        elif label == 'NULL':
            if np.random.random_sample() < 1.1:
                x.append(data)
                y.append(0)
        else:
            x.append(data)
            y.append(0)
            
    x = 1*np.array(x)
    y = np.array(y)
    
    print(x.shape)
    print(y.shape)
    
    #{0:1, 1:100}
    if event_name == 'P001':
        print('rbf channel is applied')
        model = SVC(kernel='rbf',
                    probability=True,
                    class_weight='balanced').fit(x, y)
    elif event_name == 'P002':
        print('rbf channel is applied')
        model = SVC(kernel='rbf',
                    probability=True,
                    class_weight='balanced').fit(x, y)
    elif event_name == 'P003':
        print('rbf channel is applied')
        model = SVC(kernel='rbf',
                    probability=True,
                    class_weight='balanced').fit(x, y) # best map: laplacian with (1, 98)
        #model = XGBClassifier(learning_rate=0.0001, max_depth=5, scale_pos_weight=210, max_delta_step=5)
        #model.fit(x,y)

    cPickle.dump(model, open(output_file, 'wb'), 0)
    print('SVM trained successfully for event %s!' % (event_name))
