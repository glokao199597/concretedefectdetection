 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:45:24 2020

@author: Ching
"""
#double brackets return dataframes

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.layers import *

from keras.applications import *
from keras.layers.core import Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from sklearn.metrics import accuracy_score
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import *
from keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, History
from PIL import Image
from PIL import ImageFilter
from skimage.io import imshow
from skimage.util import random_noise
from sklearn.utils import class_weight
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, precision_recall_curve,recall_score,average_precision_score
import segmentation_models as sm
import skimage.transform  
from sklearn.cluster import KMeans


import albumentations
from ImageDataAugmentor.image_data_augmentor import *
import json

keras.backend.set_image_data_format('channels_last')
global defeature, enfeature
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def my_metrics(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    true_neg = K.sum(false_pos * false_neg)
    recall = true_pos/(true_pos+false_neg)
    precision = true_pos/(true_pos+false_pos)
    f1score = (2*recall*precision)/(precision+recall)
    return recall, precision, f1score


def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def inverse_trasform(train_data,X_mean,X_std):
    new = train_data*(np.array(X_std))
    new = new+(np.array(X_mean))
    #X_train = X_train/255.0
    '''new = train_data*np.array([0.229, 0.224, 0.225])
    new = new + np.array([0.485, 0.456, 0.406])'''
    new = new.astype(np.float32)
    return new

def tversky(y_true, y_pred, smooth=1e-6, alpha=0.9):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + 0.8 * false_neg + 0.2 * false_pos + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), 1)

    
def spatial_attention(decoder_feature,encoder_feature):
    
    kernel_size = 3
    channel = decoder_feature._keras_shape[-1]
    cbam_feature = decoder_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        defeature = Permute((3, 1, 2))(cbam_feature)
    else:
        defeature = cbam_feature

    channel = encoder_feature._keras_shape[-1]
    cbam_feature = encoder_feature
   
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        enfeature = Permute((3, 1, 2))(cbam_feature)
    else:
        enfeature = cbam_feature

    ans = multiply([defeature, enfeature])
    return ans
    
X_train_path = 'Pos/'
Y_train_path = 'masks/'


image_number = next(os.walk(X_train_path))[2]
image_number.sort()
mask_number = next(os.walk(Y_train_path))[2]
mask_number.sort()
print(len(image_number))
print(len(mask_number))
X = np.zeros((len(image_number),224,224,3),dtype=np.int32)
Y = np.zeros((len(mask_number),224,224,1),dtype=np.int32)

for n, image in enumerate(image_number, 0):
    temp_img = cv2.imread(X_train_path + image,1)
    temp_img = cv2.resize(temp_img,(224,224),interpolation = cv2.INTER_LINEAR)
    X[n] = temp_img

for n, image in enumerate(mask_number, 0):
    temp_img = cv2.imread(Y_train_path + image,0)
    temp_img = cv2.resize(temp_img,(224,224),interpolation = cv2.INTER_LINEAR)
    temp_img = np.reshape(temp_img,(224,224,-1))
    ret,thresh1 = cv2.threshold(temp_img,127,255,cv2.THRESH_BINARY)
    Y[n] = temp_img/255

ori = X


x_tra, x_val, y_tra, y_val = train_test_split(X, Y, test_size = 0.1, shuffle=False)
x_tra, x_tst, y_tra, y_tst = train_test_split(x_tra, y_tra, test_size = 0.1,  shuffle=False)

#new = inverse_trasform(X,X_mean,X_std)
Aug = albumentations.Compose([
            albumentations.OneOf([               
                  albumentations.HorizontalFlip(p=1)])],p=1)

training_datagen = ImageDataAugmentor()
mask_datagen = ImageDataAugmentor()

#Aug = albumentations.Compose([])


validation_training_datagen = ImageDataAugmentor()
validation_mask_datagen = ImageDataAugmentor()

testing_datagen = ImageDataAugmentor()
print(x_tra.shape)
print(x_tst.shape)
print(x_tst[0].shape)
print(y_tst[0].shape)
inputs = Input((None,None,3))
#s = Lambda(lambda x:x/255)(inputs) #normalization

c1 = Conv2D(16,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(inputs)
c1 = Conv2D(16,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
p1 = MaxPooling2D((2,2))(c1)

c2 = Conv2D(32,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
c2 = Conv2D(32,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
p2 = MaxPooling2D((2,2))(c2)

c3 = Conv2D(64,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
c3 = Conv2D(64,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
p3 = MaxPooling2D((2,2))(c3)

c4 = Conv2D(128,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
c4 = Conv2D(128,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
p4 = MaxPooling2D((2,2))(c4)

c5 = Conv2D(256,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)

c5 = Conv2D(256,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)


u6 = Conv2DTranspose(128,(4,4),strides=(2, 2), padding = 'same')(c5)
new_c4 = spatial_attention(c4,u6)
u6 = concatenate([new_c4,u6])
c6 = Conv2D(128,(3,3), kernel_initializer = 'he_normal', padding = 'same')(u6)
c6 = BatchNormalization()(c6)
c6 = Activation('relu')(c6)
c6 = Conv2D(128,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c6)

u7 = Conv2DTranspose(64,(4,4),strides=(2, 2), padding = 'same')(c6)
new_c3 = spatial_attention(c3,u7)
u7 = concatenate([u7,new_c3])
c7 = Conv2D(64,(3,3), kernel_initializer = 'he_normal', padding = 'same')(u7)
c7 = BatchNormalization()(c7)
c7 = Activation('relu')(c7)
c7 = Conv2D(64,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)

u8 = Conv2DTranspose(32,(4,4),strides=(2, 2), padding = 'same')(c7)
new_c2 = spatial_attention(c2,u8)
u8 = concatenate([u8,new_c2])
c8 = Conv2D(32,(3,3), kernel_initializer = 'he_normal', padding = 'same')(u8)
c8 = BatchNormalization()(c8)
c8 = Activation('relu')(c8)
c8 = Conv2D(32,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)

u9 = Conv2DTranspose(16,(4,4),strides=(2, 2), padding = 'same')(c8)
new_c1 = spatial_attention(c1,u9)
u9 = concatenate([u9,new_c1], axis =3)
c9 = Conv2D(16,(3,3), kernel_initializer = 'he_normal', padding = 'same')(u9)
c9 = BatchNormalization()(c9)
c9 = Activation('relu')(c9)
c9 = Conv2D(16,(3,3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)

#final = concatenate([c9,c8,c7,c6])
c6 = UpSampling2D(size=(8,8))(c6)
c7 = UpSampling2D(size=(4,4))(c7)
c8 = UpSampling2D(size=(2,2))(c8)
final = concatenate([c9,c8,c7,c6])
outputs = Conv2D(1,(1,1),activation = 'sigmoid', kernel_initializer = 'glorot_uniform', padding = 'same')(final)

###################
model = Model(inputs=[inputs],outputs=[outputs])
model.compile(Adam(lr=3e-4),loss=focal_tversky_loss, metrics=[tversky])



image_data_augmentator = training_datagen.flow(x_tra, batch_size=16, shuffle=False)
mask_data_augmentator = mask_datagen.flow(y_tra,batch_size=16,shuffle=False)

val_image_data_augmentator = validation_training_datagen.flow(x_val, batch_size=16, shuffle=False)
val_mask_data_augmentator = validation_mask_datagen.flow(y_val,batch_size=16,shuffle=False)
#true_y = X2_.astype(np.uint8)
#true_y = true_y.astype(np.float32)

training_data_generator = zip(image_data_augmentator, mask_data_augmentator)
val_training_data_generator = zip(val_image_data_augmentator, val_mask_data_augmentator)
testing_data_augmentator = testing_datagen.flow(x_tst, batch_size=8, shuffle=False)
results_myunet = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=30, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)

preds_train = model.predict_generator(testing_data_augmentator, verbose = 1)
maxval = np.amax(preds_train[0])
print(maxval)
preds = (preds_train > 0.5).astype(np.float32)
ori = x_tst
ori_str = 'ori_'
counter = 0
for i in ori:
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(i)
    plt.savefig('Ori' + '_%d.png' %(counter))
    counter = counter + 1
    plt.close()
    # plot generted target image
counter = 0
for i in y_tst:
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(i,cmap='gray')
    plt.savefig('Tst' + '_%d.png' %(counter))
    counter = counter + 1
    plt.close()
counter = 0
for i in preds:
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(i,cmap='gray')
    plt.savefig('Pred' + '_%d.png' %(counter))
    counter = counter + 1
    plt.close()



'''model = sm.Unet('resnet34', classes = 1, activation='sigmoid', encoder_weights='imagenet')
model.compile(Adam(lr=3e-4),loss='binary_crossentropy', metrics =[dsc])
results_unet = model.fit_generator(training_data_generator, steps_per_epoch=len(x_tra)//8, epochs=30, validation_data=val_training_data_generator, validation_steps=len(x_val)//8)

preds_train = model.predict_generator(testing_data_augmentator, verbose = 1)
maxval = np.amax(preds_train[0])
print(maxval)
preds = (preds_train > 0.5).astype(np.float32)

for i in preds:
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(ori[i])
    filename1 = 'concreteseg1_unet.png'
    plt.savefig('Ori' + '_%d.png' %(counter))
    plt.close()
    # plot generted target image
for i in range(4): 
    plt.subplot(3, 4, 1 + 4 + i)
    plt.axis('off')
    plt.imshow(y_tst[i],cmap='gray')
    filename1 = 'concreteseg1_unet.png'
    plt.savefig(filename1)
    plt.close()
    # plot real target image
for i in range(4):
    plt.subplot(3, 4, 1 + 4*2 + i)
    plt.axis('off')
    plt.imshow(preds[i],cmap='gray')
    filename1 = 'concreteseg1_unet.png'
    plt.savefig(filename1)
    plt.close()
idx = 0
for i in range(5,9):
    plt.subplot(3, 4, 1 + idx)
    plt.axis('off')
    plt.imshow(ori[i])
    idx = idx+1
    # plot generted target image
idx = 0
for i in range(5,9):
    plt.subplot(3, 4, 1 + 4 + idx)
    plt.axis('off')
    plt.imshow(y_tst[i],cmap='gray')
    idx = idx+1
    # plot real target image
idx = 0
for i in range(5,9):
    plt.subplot(3, 4, 1 + 4*2 + idx)
    plt.axis('off')
    plt.imshow(preds[i],cmap='gray')
    idx = idx+1

filename1 = 'concreteseg2_unet.png'
plt.savefig(filename1)
plt.close()
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(results_unet.history['loss'], 'r', label='train')
ax.plot(results_unet.history['val_loss'], 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
fig.savefig('concreteseg_loss_unet.png')
plt.clf()



true_y = y_tst.astype(np.uint8)
true_y = true_y.astype(np.float32)
testAcc = K.get_session().run(dsc(true_y, preds))
print('DSC: ', testAcc)
recall, precision, f1score = K.get_session().run(my_metrics(true_y,preds))
print("Precision : ", precision)
print("f1score : ", f1score)
print("Recall : ", recall)'''
