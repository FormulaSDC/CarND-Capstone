# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:14:20 2017

@author: avarfolomeev
"""

import cv2
import os

input_dir = '/media/D/DIZ/out/tl/'



filelist = os.listdir(input_dir)
cascade_name = '../data/c16x32w30d2_3.xml'
cascade = cv2.CascadeClassifier(cascade_name)


for filename in filelist:
    filepath = os.path.join(input_dir,filename)
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    
    res = cascade.detectMultiScale2(gray, 1.01, 1, 0, (16,32), (100,200))
    detected = res[0];
    for result in detected:
        p0 = (result[0], result[1])
        p1 = (p0[0]+result[2], p0[1]+result[3])
        cv2.rectangle(img, p0, p1, (0,0,255),2)
        
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
    cv2.imshow("detected",img)    
    cv2.waitKey(20)
cv2.destroyWindow('detected')

