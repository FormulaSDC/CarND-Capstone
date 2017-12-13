"""
MixNet Architecture


    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
#%%
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import batch_norm

save_file = './mixNetI.ckpt'


# MixNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
def MixNet(x):
    s = 0.1
    
    #32x16x3 -> 30x14x4
    w11 = tf.Variable(tf.truncated_normal((3,3,3,4),0,s),'w11')
    b11 = tf.Variable(tf.truncated_normal([4],0,0.001),'b11')

    c1 = tf.nn.conv2d(x,w11, strides = [1,1,1,1], padding='VALID') + b11
    c1 = tf.nn.relu(c1)

    
    #30x14x4 -> 28x12x4
    w12 = tf.Variable(tf.truncated_normal((3,3,4,4),0,s),'w12')
    b12 = tf.Variable(tf.truncated_normal([4],0,0.001),'b12')
    
    c1 = tf.nn.conv2d(c1,w12, strides = [1,1,1,1], padding='VALID') + b12
    
    #28x12x8 -> 14x6x4
    c1p = tf.nn.max_pool(c1, (1,2,2,1), (1,2,2,1), padding='VALID')
    #28x12x8 -> 14x6x4
    w13 = tf.Variable(tf.truncated_normal((3,3,4,4),0,s),'w13')
    b13 = tf.Variable(tf.truncated_normal([4],0,0.001),'b13')
    c1c = tf.nn.conv2d(c1,w13, strides = [1,2,2,1], padding='SAME') + b13
    
    c1 = tf.concat([c1p,c1c], 3)

    c1 = tf.nn.relu(c1)

    ac1 = tf.nn.avg_pool(c1,(1,4,4,1),(1,4,4,1), padding='SAME')
    flat1 = flatten(ac1);    
    print("layer1 :",c1.get_shape(),ac1.get_shape(),"; flattened=", flat1.get_shape())
    #c1 = tf.nn.dropout(c1, keep_prob);
    
    #14x6x8 -> 12x4x8
    w21 = tf.Variable(tf.truncated_normal((3,3,8,8),0,s),'w21')
    b21 = tf.Variable(tf.truncated_normal([8],0,0.001),'b21')

    c2 = tf.nn.conv2d(c1,w21, strides = [1,1,1,1], padding='VALID') + b21
    c2 = tf.nn.relu(c2)
    
    
    #12x4x8 -> 10x2x8
    w22 = tf.Variable(tf.truncated_normal((3,3,8,8),0,s),'w22')
    b22 = tf.Variable(tf.truncated_normal([8],0,0.01),'b22')
    
    c2 = tf.nn.conv2d(c2,w22, strides = [1,1,1,1], padding='VALID') + b22
    
    #10x2x8 -> 5x1x8
    c2p = tf.nn.max_pool(c2, (1,2,2,1), (1,2,2,1), padding='VALID')
    #10x2x8 -> 5x1x8
    w23 = tf.Variable(tf.truncated_normal((3,3,8,8),0,s),'w23')
    b23 = tf.Variable(tf.truncated_normal([8],0,0.01),'b23')
    c2c = tf.nn.conv2d(c2,w23, strides = [1,2,2,1], padding='SAME') + b23

    #5x1x8 + 5x1x8 = 5x1x16
    c2 = tf.concat([c2p,c2c], 3)

    #5X1X16->80
    flat2 = flatten(c2);    
    print("layer2 :",c2.get_shape(),"; flattened=", flat2.get_shape())
    
    
   
    #400+100
    lin1 = tf.concat([flat1, flat2], 1)
    lin1len = int(lin1.get_shape()[1]);
    print("lin1 shape:",lin1.get_shape(), lin1len)

    lin1 = tf.nn.tanh(lin1)

    #500 -> 100
    wl1 = tf.Variable(tf.truncated_normal((lin1len,20),0,s),'wl1')
    bl1 = tf.Variable(tf.truncated_normal([20],0,0.001),'bl1')

    lin1 = tf.nn.dropout(lin1, keep_prob)

    lin1 = tf.matmul(lin1,wl1) + bl1
    lin1 = tf.nn.tanh(lin1, name = 'tanh1')

    lin1 = tf.nn.dropout(lin1, keep_prob)
    
    wl2 = tf.Variable(tf.truncated_normal((20,4),0,s),'wl2')
    bl2 = tf.Variable(tf.truncated_normal([4],0,0.001),'bl2')

    lin2 = tf.matmul(lin1,wl2) + bl2
    
    out = tf.identity(lin2, name = 'out')
   
    return out;
    
