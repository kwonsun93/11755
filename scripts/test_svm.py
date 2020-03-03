#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
from sklearn.metrics.pairwise import chi2_kernel
from xgboost import XGBClassifier
import _pickle as cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
        print("model_file -- path of the trained svm file")
        print("feat_dir -- dir of feature files")
        print("feat_dim -- dim of features; provided just for debugging")
        print("output_file -- path to save the prediction score")
        exit(1)

    model_file = sys.argv[1]     # mfcc_pred/svm.$event.model
    feat_dir = sys.argv[2]       # ./kmeans/ or ./asrfeat/
    feat_dim = int(sys.argv[3])  # 200 or 970
    output_file = sys.argv[4]    # mfcc_pred/${event}_mfcc.lst
    event_name = model_file.split('.')[-2]
    
    fwrite = open(output_file, 'w')

    # data loading
    test_list = open('../all_val.lst', 'r')
    x = []
    for file in test_list.readlines():
        filename = file[:-1].split(' ')[0]
        data_path = feat_dir + filename + '.feats'
        data = np.genfromtxt(data_path, delimiter=';')
        
        x.append(data)            
    x = 1*np.array(x)
    
    # Prediction
    model = cPickle.load(open(model_file, 'rb'))
    pred = model.predict(x)
    print(pred)
    
    # Writing results
    for p in pred:
        fwrite.write(str(p) + '\n')
    fwrite.close()
    
