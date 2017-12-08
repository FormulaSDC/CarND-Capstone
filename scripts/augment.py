# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:12:47 2017

@author: avarfolomeev
"""

import pickle
import numpy as np
import math
import cv2
import os



import math

def getMotionKernel(size):
    kernel = np.zeros((size, size));
    kernel[int((size-1)/2), :] = np.ones(size)/size;
    return kernel
    
motion_kern3 = getMotionKernel(3);
motion_kern5 = getMotionKernel(5);    
    
def getPerspMatrix(x, y, z, size):
    w, h = size;
    half_w = w/2.;
    half_h = h/2.;

    
    rx = math.radians(x);
    ry = math.radians(y);
    rz = math.radians(z);
    
    cos_x = math.cos(rx);
    sin_x = math.sin(rx);
    cos_y = math.cos(ry);
    sin_y = math.sin(ry);
    cos_z = math.cos(rz);
    sin_z = math.sin(rz);
 
     # Rotation matrix:
    # | cos(y)*cos(z)                       -cos(y)*sin(z)                     sin(y)         0 |
    # | cos(x)*sin(z)+cos(z)*sin(x)*sin(y)  cos(x)*cos(z)-sin(x)*sin(y)*sin(z) -cos(y)*sin(y) 0 |
    # | sin(x)*sin(z)-cos(x)*sin(y)*sin(z)  sin(x)*sin(z)+cos(x)*sin(y)*sin(z) cos(x)*cos(y)  0 |
    # | 0                                   0                                  0              1 |

    R = np.float32(
        [
            [cos_y * cos_z,  cos_x * sin_z + cos_z * sin_y * sin_x],
            [-cos_y * sin_z, cos_z * cos_x - sin_z * sin_y * sin_x],
            [sin_y,          cos_y * sin_x],
        ]
    );

    center = np.float32([half_h, half_w]);
    offset = np.float32(
        [
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h],
        ]
    );

    points_z = np.dot(offset, R[2]);
    dev_z = np.vstack([w/(w + points_z), h/(h + points_z)]);

    new_points = np.dot(offset, R[:2].T) * dev_z.T + center;
    in_pt = np.float32([[0, 0], [w, 0], [w, h], [0, h]]);

    transform = cv2.getPerspectiveTransform(in_pt, new_points);
    return transform;



def transformImg(img, x=0, y=0, z=0, scale = 1):
    size = img.shape[:2][::-1]
    
    M = getPerspMatrix(x, y, z, size)
    if scale != 1:
        S = np.eye(3);
        S[0,0] = S[1,1] = scale;
        S[0,2] = size[0]/2 * (1-scale);
        S[1,2] = size[1]/2 * (1-scale);
        M = np.matmul(S,M);

    result = cv2.warpPerspective(img, M, size, borderMode=cv2.BORDER_REPLICATE)

    return result

    
                             
#replicate img N times
def augmentImage(img, N:int):
    
    out = [img];

    rangeX = [  0, 0.1];
    rangeY = [-0.1, 0.1];
    rangeZ = [-0.1, 0.1];
    rangeS = [0.95, 1.05]
    rangeI = [-80, 80];


    for i in range(N-1):
        x = np.random.uniform(rangeX[0], rangeX[1]);
        y = np.random.uniform(rangeY[0], rangeY[1]);
        z = np.random.uniform(rangeZ[0], rangeZ[1]);
        scale = np.random.uniform(rangeS[0], rangeS[1]);
        motion = np.random.uniform();
        if motion > 0.95:
            tmp = cv2.filter2D(img,-1,motion_kern5);
        elif motion > 0.5 :
            tmp = cv2.filter2D(img,-1,motion_kern3);
        else:
            tmp = img;
        intens = np.random.uniform(rangeI[0], rangeI[1]);
        img_mean = np.mean(tmp[8:24,4:12]);
        if (intens + img_mean > 240):
            intens = 240-img_mean;
        elif (intens + img_mean < 10):
            intens = 10-img_mean;
        intens = np.ones(4) * intens    
        tmp = cv2.add(tmp, intens,dtype=0);
        out.append(tmp); #transformImg(tmp,x,y,z,scale));
    return out;
#%%

        
def normalizeImage0(img):
        imf = np.float32(img);
        imf = (imf - 127.)/255.;
        return imf;
        
        
#%%    

def normalizeImageList(imgList, mode = '0'):
    print('Constant normalization');
    out = np.array([normalizeImage0(img) for img in imgList]);
    return out;
    
    
#%%    
#build new list of N images    
def augmentImgClass(imgList, outOrN ):
    inputLen = len(imgList);
    shape = list(imgList[0].shape);
    if (type(outOrN) == int):
        outLen = outOrN;
        if (outLen < inputLen):
            outLen = inputLen
        shape.insert(0, outLen)  #to form output array
        out = np.empty(shape, np.uint8)
    elif (type(outOrN)==np.ndarray):
        out = outOrN;
        outLen = out.shape[0];
    else:
        print("invalid second argument")
        return np.empty(0)

    #print("augmenting ", inputLen, "images to", outLen);        
        

    k = 0
    l = 0
    for  img in imgList:
        #img = normalizeImage0(img)
        cf = np.int((outLen-k)/(inputLen-l)) + 1;
        if (cf > 1):
            newImages = augmentImage(img, cf);
            print("augmentImgClass: ", img.shape, newImages[0].shape, out[0].shape)

            l = l+1;
            for imNew in newImages:
                if (k < outLen):
                    out[k]=imNew;
                k = k+1;
        else:
            if (k < outLen):
                out[k] = img;
            k = k+1;
    #print (l,k,cf)
    return out;
        
                                
#%%

def augmentImageSet(Set,targetCount):

    classes = list(Set.keys());
    
    out = dict();
    #totalLen = targetCount * nClasses;
    #targetXShape = list(Set[classes[0]][0].shape);
    #targetXShape.insert(0, totalLen); 
    
    #targetX = np.empty(targetXShape,dtype = np.float32);
    #targetY = np.empty(totalLen, dtype = np.uint8);
                     
    for signClass in classes:
        print("filling class ", signClass);
        inputImages = Set[signClass];
        out[signClass] = augmentImgClass(inputImages, targetCount);
        #targetY[signClassNr*targetCount:(signClassNr+1)*targetCount] = signClassNr;

    #idx = np.arange(totalLen);
    #np.random.shuffle(idx);
        #shuffle
    #targetY = targetY[idx];
    #targetX = targetX[idx];
    
    return out;


#%%
def SplitSet(total,pTst, pVal, minCnt=1):
    
    dTrn = dict()
    dVal = dict()
    dTst = dict()
    
    classList = list(total.keys());
    
    for cls in classList:
        n = len(total[cls]);
        i = np.arange(n)
        np.random.shuffle(i)
        
        nTst = int (n*pTst)
        if (nTst < minCnt):
            nTst = minCnt
        nVal = int (n*pVal)
        if (nVal < minCnt):
            nVal = minCnt;
            
        print("Splitting class", cls, n, ':', n-nTst-nVal, nVal, nTst)    
        dTst[cls] = total[cls][:nTst]
        dVal[cls] = total[cls][nTst:nTst+nVal]
        dTrn[cls] = total[cls][nTst+nVal:]

    return {"trn":dTrn, "val":dVal, "tst":dTst}

#%%
def writeSet(Set, outDir):
    if (not os.path.exists(outDir)):
        os.mkdir(outDir)
    for cls in Set.keys():
        print(cls)
        cdir = os.path.join(outDir,cls)
        if (not os.path.exists(cdir)):
            os.mkdir(cdir)
        cnt = 1
        for img in Set[cls]:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            name = '%s_%04d.png' % (cls, cnt)
            cv2.imwrite(os.path.join(cdir,name), img)
            cnt = cnt+1
            
            
    
            
                