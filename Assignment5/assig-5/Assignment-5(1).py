#!/usr/bin/env python
# coding: utf-8

# In[50]:


import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from sklearn.model_selection import train_test_split


# In[51]:


((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.fashion_mnist.load_data()


# In[ ]:


"""Data Augmentation using opencv"""


# In[56]:


def translation_image(image):
    rows, cols ,c= image.shape
    M = np.float32([[1, 0, -5], [0, 1, -5]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image
    
def rotate_image(image):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 30, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def rotate_image1(image):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def flip_image_vertical(image):
    rows,cols,c = image.shape
    image= cv2.flip(image, 0)
    return image

def flip_image_horizontal(image):
    rows,cols,c = image.shape
    image= cv2.flip(image, 1)
    return image

def flip_image_both(image):
    rows,cols,c = image.shape
    image= cv2.flip(image, -1)
    return image


# In[57]:


def Augmentation(img, n):
    if(n==0):
        return translation_image(img)
    elif(n==1):
            return rotate_image(img)
    elif(n==2):
            return rotate_image1(img)
    elif(n==3):
            return flip_image_vertical(img)
    elif(n==4):
            return flip_image_horizontal(img)
    elif(n==5):
            return flip_image_both(img)
    else:
        return img


# In[58]:


Title = ["translation_image","rotate_image","rotate_image1","flip_image_vertical","flip_image_horizontal","flip_image_both"]


# In[59]:


plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
plt.imshow(train_data[0,:,:])
plt.xticks([])
plt.yticks([])
plt.title("Original")

for i in range(6):
    plt.subplot(2,4,i+2)
    imx = Augmentation(train_data[0,:,:].reshape(28,28,1),i)
    plt.imshow(imx.reshape(28,28))
    plt.xticks([])
    plt.yticks([])
    plt.title(Title[i])


# In[31]:


target_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}


# In[32]:


print(train_data.shape)
print(eval_data.shape)


# In[33]:


plt.figure(figsize=(10,10))
for i in range(0,5):
    plt.subplot(5,5, i+1)
    plt.imshow(train_data[i] )
    plt.title( target_dict[(train_labels[i]) ])
    plt.xticks([])
    plt.yticks([])


# In[34]:


train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)


# In[35]:


def cnn_model(features, labels, mode):
    #Reshapinng the input
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
     # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    
    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[5, 5],
        
          padding="same",
          activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
       
    flatten_1= tf.reshape(pool3, [-1, 3*3*128])
    
    dense = tf.layers.dense(inputs= flatten_1,units=1024,activation=tf.nn.relu)
    
    
    output_layer = tf.layers.dense(inputs= dense, units=10)
    predictions={
    "classes":tf.argmax(input=output_layer, axis=1),
    "probabilities":tf.nn.softmax(output_layer,name='softmax_tensor')
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss= tf.losses.sparse_softmax_cross_entropy(labels=labels, logits= output_layer, scope='loss')
    
    if mode== tf.estimator.ModeKeys.TRAIN:
        optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
        train_op= optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op )
    
    eval_metrics_op={ "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)


# In[36]:


fashion_classifier = tf.estimator.Estimator(model_fn = cnn_model)


# In[37]:


# Training the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

fashion_classifier.train(input_fn=train_input_fn, steps=1000)


# In[38]:


#Evaluation using test data
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = fashion_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


# In[47]:


********************************CNN Model with dropout*******************************************************************


# In[39]:


def cnn_model_dp(features, labels, mode):
    #Reshapinng the input
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
     # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    dropout_1 = tf.layers.dropout(inputs=pool2, rate=0.25,training=mode == tf.estimator.ModeKeys.TRAIN )
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv3 = tf.layers.conv2d(
          inputs=dropout_1,
          filters=128,
          kernel_size=[3, 3],
        
          padding="same",
          activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
    dropout_2 = tf.layers.dropout(inputs=pool3, rate=0.25,training=mode == tf.estimator.ModeKeys.TRAIN )
       
    flatten_1= tf.reshape(dropout_2, [-1, 3*3*128])
    
    dense = tf.layers.dense(inputs= flatten_1,units=1024,activation=tf.nn.relu)
    
    dropout= tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    output_layer = tf.layers.dense(inputs= dropout, units=10)
    predictions={
    "classes":tf.argmax(input=output_layer, axis=1),
    "probabilities":tf.nn.softmax(output_layer,name='softmax_tensor')
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss= tf.losses.sparse_softmax_cross_entropy(labels=labels, logits= output_layer, scope='loss')
    
    if mode== tf.estimator.ModeKeys.TRAIN:
        optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
        train_op= optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op )
    
    eval_metrics_op={ "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)


# In[40]:


fashion_classifier_dp = tf.estimator.Estimator(model_fn = cnn_model_dp)


# In[41]:


# Training of the model
train_input_fn_dp = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

fashion_classifier_dp.train(input_fn=train_input_fn_dp, steps=1500)


# In[42]:


#Evaluation using test data
eval_input_fn_dp = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results_dp = fashion_classifier_dp.evaluate(input_fn=eval_input_fn_dp)
print(eval_results_dp)


# In[ ]:


******************************CNN Model with batch normalization********************************************************


# In[43]:



def cnn_model_batchnorm(features, labels, mode):
    #Reshapinng the input
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
     # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same")
    conv1= tf.layers.batch_normalization(conv1)
    conv1=tf.nn.relu(conv1)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same")
    conv2= tf.layers.batch_normalization(conv2)
    conv2=tf.nn.relu(conv2)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    
    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[5, 5],
          padding="same")
    conv3= tf.layers.batch_normalization(conv3)
    conv3=tf.nn.relu(conv3)
    
    
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
       
    flatten_1= tf.reshape(pool3, [-1, 3*3*128])
    
    dense = tf.layers.dense(inputs= flatten_1,units=1024)
    dense = tf.layers.batch_normalization(dense)
    dense=tf.nn.relu(dense)
    
    
    output_layer = tf.layers.dense(inputs= dense, units=10)
    predictions={
    "classes":tf.argmax(input=output_layer, axis=1),
    "probabilities":tf.nn.softmax(output_layer,name='softmax_tensor')
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss= tf.losses.sparse_softmax_cross_entropy(labels=labels, logits= output_layer, scope='loss')
    
    if mode== tf.estimator.ModeKeys.TRAIN:
        optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
        train_op= optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op )
    
    eval_metrics_op={ "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)


# In[44]:


fashion_classifier_batchnorm = tf.estimator.Estimator(model_fn = cnn_model_batchnorm)


# In[45]:


# Train the model
train_input_fn_batchnorm = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

fashion_classifier_batchnorm.train(input_fn=train_input_fn_batchnorm, steps=1500)


# In[46]:


#Evaluation using test data
eval_input_fn_batchnorm = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results_batchnorm = fashion_classifier_batchnorm.evaluate(input_fn=eval_input_fn_batchnorm)
print(eval_results_batchnorm)


# In[ ]:




