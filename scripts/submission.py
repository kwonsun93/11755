import os
import numpy as np
import pandas as pd
import _pickle as cPickle

# data loading
test_list = open('/home/ubuntu/11775-hws/all_test.video', 'r')
feat_dir = '../soundnetfeat/'
x = []
filenames = []
for file in test_list.readlines():
    filename = file[:-1]
    filenames.append(filename)
    data_path = feat_dir + filename + '.feats'
    data = np.genfromtxt(data_path, delimiter=';')
    x.append(data)            
x = 1*np.array(x)

# Prediction
events = ['P001','P002','P003']
preds = []
for event in events:
    model_path = '../soundnet_pred/svm.' + event + '.model'
    print(model_path)
    model = cPickle.load(open(model_path, 'rb'))
    pred = model.decision_function(x)
    #pred = model.predict_proba(x)[:, 1]
    preds.append(pred)
    print(pred)
    print("=======")
    
preds = np.array(preds)
final = np.zeros(preds.shape[1])
for i in range(preds.shape[1]):
    a = preds[:, i]
    if max(a) > -0.75:
        final[i] = np.argmax(a) + 1
    else:
        final[i] = 1
final = final.astype(int)

# Make submission file
df = pd.DataFrame(data={"VideoID": filenames, "Label": final})
df.to_csv("/home/ubuntu/11775-hws/hw1_code/SampleSubmission.csv", sep=',',index=False)
