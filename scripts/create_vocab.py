import numpy as np
import glob
import os

# save directory
save_path = '/home/ubuntu/11775-hws/vocab/vocab'
fwrite = open(save_path, 'w')

# wordcount
wordcount = {}

# Build wordcoutn
data_dir = '/home/ubuntu/11775-hws/asrs/*.txt'
step = 0
count= 0
#print(glob.glob(data_dir))
for filename in glob.glob(data_dir):
    fread = open(filename, 'r')
    
    file_dict = []
    for line in fread.readlines():
        line = line[:-2].lower() # remove '.' and '\n'
        ws = line.split(' ')
        for w in ws:
            if w not in file_dict:
                file_dict.append(w)
                wordcount[w] = 1
            else:
                wordcount[w] += 1
    fread.close()
    
cc = 0
for k, v in wordcount.items():
    if v > 1 and v < 2000:
        if "'" not in k and len(k) > 2:
            fwrite.write(k + '\n')
            print(k, v)
            cc += 1
        
