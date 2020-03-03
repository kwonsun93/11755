#!/bin/python
import numpy as np
import os
import _pickle as cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print("kmeans_model -- path to the kmeans model")
        print("cluster_num -- number of cluster")
        print("file_list -- the list of videos")
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    
    fread = open(file_list, "r")
    
    for line in fread.readlines():
        mfcc_path = "mfcc/" + line.replace('\n','') + ".mfcc.csv"
        out_path = "kmeans/" + line.replace('\n','') + ".feats"
        fwrite = open(out_path, 'w')
        
        if os.path.exists(mfcc_path) == False:
            #bow = np.zeros(cluster_num)
            bow = np.ones(cluster_num) / cluster_num
        else:
            bow = np.zeros(cluster_num)
            array = np.genfromtxt(mfcc_path, delimiter=";")
            classes = kmeans.predict(array)
            
            for c in classes:
                bow[c] += 1
            bow = bow / len(classes)
            
        line = str(bow[0])
        for m in range(1, cluster_num):
            line += ';' + str(bow[m])
        fwrite.write(line + '\n')
        fwrite.close()
           
    print("K-means features generated successfully!")
