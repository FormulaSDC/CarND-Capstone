# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:25:03 2017

@author: avarfolomeev
"""
import os
import math
import random
import time
import shutil
import itertools
import multiprocessing as mp
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import cv2

from augment import *

#%%
#baseDir = 'F:/Datasets/Udacity/TL-1'
baseDir = '/media/D/DIZ/CarND/capstone/my code/scripts'

#%%
classes = os.listdir(baseDir)

imgList = []
yN = []
yStr = []
nClasses = len(classes)
dClasses = dict()


for c in range(nClasses):
    cls = classes[c]
    clPath = os.path.join(baseDir, cls)
    files = os.listdir(clPath)
    print(c, 'from ', nClasses, cls, " : ", len(files))
    
    cImgList = [] #images for this class
    for f in files:
        img = cv2.imread(os.path.join(clPath,f))
        if (img.shape[2] == 3):
            cv2.cvtColor(img,cv2.COLOR_BGR2RGB, img)
        else:
            cv2.cvtColor(img,cv2.COLOR_BGRA2RGB, img)
        w = img.shape[1]
        img = cv2.resize(img,(16,32))
        cImgList.append(img)
        #yStr.append(cls)
        yN.append(c)
    dClasses[cls] = cImgList
    #imgList.extend(cImgList)
        

    
#%%
import pickle
write = False
if (write):
    pickle.dump(dClasses, open(baseDir+"/../dClasses-1.p",'wb'))
    dSets = SplitSet(dClasses, 0.01, 0.1, 0)
    pickle.dump(dSets, open(baseDir+"/../dSets-1.p",'wb'))
else:
    dClasses = pickle.load(open(baseDir+"/../dClasses-1.p",'rb'))
    dSets = pickle.load(open(baseDir+"/../dSets-1.p",'rb'))
    
    
#%%



writeSet(dSets['val'],'F:/Datasets/Udacity/sets/val')    
writeSet(dSets['tst'],'F:/Datasets/Udacity/sets/tst')
  
#%%
augSize = 512
dTrn = dSets['trn']
dVal = dSets['val']

trnA = augmentImageSet(dTrn,augSize)



sz = list(trnA[keys[0]].shape)
sz[0] = 0;

Y_t = []
X_t = np.zeros(sz,dtype='float32')
Y_v = []
X_v = np.zeros(sz,dtype='float32')



keys = ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']

for cls in range(4):
    Y_t = np.append(Y_t, np.zeros(augSize)+cls)
    X_t = np.append(X_t, trnA[keys[cls]],0)
    Y_v = np.append(Y_v, np.zeros(len(dVal[keys[cls]]))+cls)
    X_v = np.append(X_v, dVal[keys[cls]],0)
    

Xgn_t = normalizeImageList(X_t)
Xgn_v = normalizeImageList(X_v)
    
S = {"Xgn_t":Xgn_t, "Y_t":Y_t, "Xgn_v":Xgn_v, "Y_v":Y_v}
pickle.dump(S, open(baseDir+"/../TrnValgn.p",'wb'))


#%%
import pickle

baseDir = '/media/D/DIZ/CarND/capstone/my code/scripts'

S = pickle.load(open(baseDir+"/../TrnValgn-1.p",'rb'))
lenTrn = len(S['Y_t'])
lenVal = len(S['Y_v'])

trnIdx = np.arange(lenTrn)
valIdx = np.arange(lenVal)
np.random.shuffle(trnIdx)
np.random.shuffle(valIdx)

Xgn_t = S['Xgn_t'][trnIdx]
Xgn_v = S['Xgn_v'][valIdx]
Y_t = S['Y_t'][trnIdx]
Y_v = S['Y_v'][valIdx]



            