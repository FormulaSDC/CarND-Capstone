# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:33:36 2017

@author: diz
"""



#
tf.reset_default_graph();
keep_prob = tf.placeholder(tf.float32, name='keep_prob')                                           
batch_x = tf.placeholder(tf.float32, [None,32,16,3], name='batch_x')
batch_y = tf.placeholder(tf.int32, (None),name='batch_y')
ohy = tf.one_hot(batch_y,4,name='one_hot');
fc2 = MixNet(batch_x)

step = tf.Variable(0, trainable=False,name='step')


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=ohy))
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.minimize(loss_op, global_step = step)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(ohy, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

e_t = tf.where(tf.not_equal(tf.cast(tf.argmax(fc2, 1),tf.int32), batch_y))
softmax = tf.nn.softmax(fc2);        
top_k = tf.nn.top_k(softmax, k=5)

saver = tf.train.Saver();


ans = tf.argmax(fc2, 1);

def test_new_data(xv, yv):
    sess = tf.get_default_session()
 
    loss, acc, err = sess.run([loss_op, accuracy_op, e_t], feed_dict={batch_x : xv , batch_y: yv, keep_prob: 1.0})

    top, result = sess.run([top_k,softmax],   feed_dict={batch_x : xv , keep_prob: 1.0})     
    return top, result , err, loss, acc



#%%
#newLabels = [14,28,8,10,4,5,5,20,31,31]

newLabels = [5,22,9,14,4,17,3,3,16,10]

print ("testing " , save_file)

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_file)


    top, result, e, val_loss, val_acc = test_new_data(newImagesN[:], newLabels[:]); #Xgn_test,y_test)
    print("Validation loss = {:.3f}".format(val_loss))
    print("Validation accuracy = {:.3f}".format(val_acc))

    # Evaluate on the test data


#%%
nImages = newImages.shape[0]


figsize = (15, 20)

gs = gridspec.GridSpec(nImages , 7)

fig1 = plt.figure(num=1, figsize=figsize)
ax = []
for i in range(nImages):
    row = (i)
    ax.append(fig1.add_subplot(gs[row, 0]))
    #example
    img = newImages[i]
    ax[-1].imshow(img)
    ax[-1].axis('off')
    
    ax.append(fig1.add_subplot(gs[row, 1]))
    ax[-1].bar(ind+0.5, top[0][i], 0.5, color='b')
    ax[-1].set_yscale('log')
    ax[-1].set_xlim((0,6))
    ax[-1].set_xticks([])
    ax[-1].set_title('log(p)')
    for t in range (top[1].shape[1]):
        ax.append(fig1.add_subplot(gs[row, 2+t]))
        #example
        img = examples[top[1][i][t]]
        ax[-1].imshow(img)
        ax[-1].set_title('%.5f' % (top[0][i][t]))
        ax[-1].axis('off')
        

        
#%%
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_file)
    a = sess.run(ans, feed_dict={batch_x: [img, img1],  keep_prob: 1.0})
        
