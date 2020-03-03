#!/bin/python
import numpy as np
import os
import _pickle as cPickle
from sklearn.cluster.k_means_ import KMeans
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} vocab_file, file_list".format(sys.argv[0]))
        print("vocab_file -- path to the vocabulary file")
        print("file_list -- the list of videos")
        exit(1)
        
    vocab_file = sys.argv[1]
    file_list = sys.argv[2]
    
    fread_vocab = open(vocab_file, 'r')
    fread = open(file_list, 'r')
    
    # load vocab
    vocab = []
    for v in fread_vocab.readlines():
        v = v.replace('\n', '')
        vocab.append(v)
    vocab_num = len(vocab)
    
    # create asr features
    for line in fread.readlines():
        txt_path = 'asr/' + line.replace('\n', '') + '.txt'
        out_path = 'asrfeat/' + line.replace('\n', '') + '.feats'
        fwrite = open(out_path, 'w')
        
        if os.path.exists(txt_path) == False:    
            #bow = np.zeros(vocab_num)
            bow = np.ones(vocab_num) / vocab_num
        else:
            bow = np.zeros(vocab_num)
            
            fread_txt = open(txt_path, 'r')
            for txtline in fread_txt.readlines():
                txtline = txtline[:-2].lower()
                ws = txtline.split(' ')
                
                for w in ws:
                    if w in vocab:
                        bow[vocab.index(w)] += 1
                        
                if np.sum(bow) == 0:
                    #bow = np.zeros(vocab_num)
                    bow = np.ones(vocab_num) / vocab_num
                else:
                    bow = bow / np.sum(bow)
            fread_txt.close()
                    
        l = str(bow[0])
        for m in range(vocab_num):
            l += ';' + str(bow[m])
        fwrite.write(l + '\n')
        fwrite.close()


    print("ASR features generated successfully!")
