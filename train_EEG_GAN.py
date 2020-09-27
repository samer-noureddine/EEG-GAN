import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ast
import numpy as np
import scipy.io
mat = scipy.io.loadmat('YOURDIRECTORY')
eeg = mat['ALL_CLEAN_EEG']



# define the generator

def generator (z, reuse =None):
    with tf.variable_scope(name_or_scope  = 'gen',reuse = reuse):
        hidden1 = tf.layers.dense(inputs = z, units = 128)
        #leaky relu
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1,hidden1)
        hidden2 = tf.layers.dense(inputs = hidden1, units = 128)
        
        hidden2 = tf.maximum(alpha*hidden2, hidden2)
        output = tf.layers.dense(inputs = hidden2, units = 1024, activation = tf.nn.tanh)
        return output


# define discriminator

def discriminator(X, reuse = None):
    '''this network is going to generate Variables (weights) implicitly I guess
        You don't want to create a NEW discriminator network every time you call this
        That is, you want to reuse the same weights and biases (which are get_variable types)
        So put reuse = True when relevant.'''
    with tf.variable_scope(name_or_scope = 'dis', reuse = reuse):
        hidden1 = tf.layers.dense(inputs = X, units = 128)
        #leaky relu
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1, hidden1)
        hidden2 = tf.layers.dense(inputs = hidden1, units = 128)
        hidden2 = tf.maximum(alpha*hidden2,hidden2)
        logits = tf.layers.dense(inputs = hidden2, units = 1)
        output = tf.sigmoid(logits)
        return output, logits

def loss_func(logits_in, labels_in):
    # this will compute  y*-log(h) + (1-y)*-log(1-h), where h = sigmoid(logits_in); y = labels_in
    # logits_in is a vector of real numbers: -inf to +inf
    # what you get is a mx10 vector; a loss value for each predicted feature for each training example
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_in, labels = labels_in))
def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]


real_images = tf.placeholder(tf.float32 ,shape =[None,1024])
z = tf.placeholder(tf.float32 ,shape =[None,100])

# Generator output
G = generator(z)

# Discriminator output
# the reason I care about logits is just to compute cross entropy loss.
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G,reuse = True)

learning_rate = 0.001
#below is D_real_loss = loss_func(logits_in = D_logits_real, labels_in = tf.ones_like(D_logits_real)*(0.9))
D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real)*(0.9))
D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_real))
D_loss = D_real_loss+ D_fake_loss
G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

# this is a convenient way to get all
# the weights, biases and whatever trainables you've made
# into one place. They're all Variable types and have .name values.
# since we gave everything in D (and everything in G) one name,
# we have them in this list and can access them separately
tvars = tf.trainable_variables()


d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]
print ([v.name for v in d_vars ])
print ([v.name for v in g_vars ])
D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list = d_vars)

G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list = g_vars )
count = 0
batch_size = 129
start, end = [],[]
# ---------------------------------------------------------
epochs = 500
import time
# ---------------------------------------------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list = g_vars)
# Save a sample per epoch
samples = []
with tf. Session () as sess:
    sess.run(init)
    d_loss = []
    g_loss = []
    count = 0
    # Recall an epoch is an entire run through the training data
    for e in range(epochs):
        print(" Currently on Epoch " + str(e+1) +' out of ' + str(epochs))
        start.append(time.time())
        for batch_images in group_list(eeg,129):
            x = batch_images
            count+=1
            # Z ( random latent noise for Generator )
            # -1 to 1 because of tanh activation
            batch_z = np.random.uniform(-1,1, size =(129, 100))
            # Run optimizers , no need to save outputs , we won ’t use them
            _ = sess.run(D_trainer , feed_dict ={real_images:batch_images, z:batch_z })
            _ = sess.run(G_trainer , feed_dict ={z:batch_z})
            g_loss.append(sess.run(G_loss, feed_dict ={z:batch_z}))
            d_loss.append(sess.run(D_loss, feed_dict ={real_images:batch_images, z:batch_z}))
        end.append(time.time())
        # Sample from generator as we ’re training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size =(1,100))
        gen_sample = sess.run(generator(z ,reuse=True),feed_dict ={z:sample_z})
        samples.append(gen_sample)
        count= count+ 1

ax = pd.DataFrame({'Generative Loss':g_loss, 'Discriminative Loss':d_loss}).plot(title ='Training loss', logy=True)
ax.set_xlabel(" Training iteration ")
ax.set_ylabel("Loss")
fig,axes = plt.subplots(6,10,figsize = (10,2))
axes[i][j].imshow(samples[num], cmap = 'Blues', interpolation = 'nearest')
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])
plt.show()
