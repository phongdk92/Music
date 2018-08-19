#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 23:06:06 2018

@author: phongdo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import subprocess
import commands
import os
import sys
import cPickle
import keras as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, LSTM, Dropout, Input, Flatten, InputLayer, Bidirectional 
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from datetime import datetime
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import pandas as pd

#from utils import normalize_data, get_new_max_length, get_same_length_data

N_FEATURES = 34
win = 0.025
step = 0.01

path = 'train'
if not os.path.exists('model'):
    os.mkdir('model')
model_name= 'model/' + datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")

#model_name = 'model/2018_08_16_09_14'

def extract_features(path_to_folder='.'):
    print 'extract_features'
    train_pkl = os.path.join(path_to_folder,'train.pkl')
    if os.path.isfile(train_pkl):
        [training_set, maxlen, max_cepstrum, min_cepstrum] = cPickle.load(open(train_pkl, 'rb'))
        return training_set, maxlen, max_cepstrum, min_cepstrum
    
    training_set = []
    max_cepstrum = np.zeros(N_FEATURES)
    min_cepstrum = np.zeros(N_FEATURES)
    maxlen = 0
    df = pd.read_csv('train.csv', header=None, names=['filename', 'catefory'])
    df.sort_values(by=['filename'])
    
    list_filenames = sorted(os.listdir(path_to_folder))
    for (i,filename) in enumerate(list_filenames):
        path_to_file = os.path.join(curDir,filename)
        [rate,sig] = audioBasicIO.readAudioFile(path_to_file)
        if (rate == -1 and sig == -1):
            #convert to wav
            #command = "ffmpeg -i {}".format(path_to_file)
            extension = os.path.splitext(filename)[-1]
            new_file = path_to_file.replace(extension, '.wav')
            command = "ffmpeg -i {} {}".format(path_to_file, new_file)
            os.system(command)
            [rate,sig] = audioBasicIO.readAudioFile(new_file)
            os.system('rm {}'.format(path_to_file))     #remove old file not in *.wav format
        if sig.ndim >= 2:           #merge multichannels into mono channel
            sig = np.mean(sig,axis=1)
#            (rate,sig) = wav.read(path_to_file)
#            mfcc_feat = mfcc(sig,rate, nfft = 2048)
        features = audioFeatureExtraction.stFeatureExtraction(sig, rate, win*rate, step*rate);
        features = features.reshape((features.shape[1],-1))
        maxlen = max(maxlen, features.shape[0])        #find max time-length
        
        max_current_mfcc = np.max(features, axis = 0)
        max_cepstrum = np.maximum(max_cepstrum, max_current_mfcc)
        
        min_current_mfcc = np.min(features, axis = 0)
        min_cepstrum = np.minimu10m(min_cepstrum, min_current_mfcc)
        #print sig.shape
        #print mfcc_feat.shape
        sample = {'feature': features,
                  'category': np.zeros(10, np.int16)}
        sample['category'][df.loc[filename].values[0]] = 1
        training_set.append(sample)
    print len(training_set)
    cPickle.dump([training_set, maxlen, max_cepstrum, min_cepstrum], open(train_pkl,'wb'), -1)
    return training_set, maxlen, max_cepstrum, min_cepstrum       
        
def split_data(training_set):
    train, test = train_test_split(training_set, test_size = 0.3, random_state=42)
    X_train = np.array([element['feature'] for element in train])
    Y_train_gender = np.array([element['gender'] for element in train])
    Y_train_accent = np.array([element['accent'] for element in train])
    assert (X_train.shape[0] == Y_train_gender.shape[0]) and (X_train.shape[0] == Y_train_accent.shape[0])
    
    X_test = np.array([element['feature'] for element in test])
    Y_test_gender = np.array([element['gender'] for element in test])
    Y_test_accent = np.array([element['accent'] for element in test])
    assert (X_test.shape[0] == Y_test_gender.shape[0]) and (X_test.shape[0] == Y_test_accent.shape[0])
    
    return X_train, Y_train_gender, Y_train_accent, X_test, Y_test_gender, Y_test_accent

def model(training_set, maxlen, use_dropout = True):
    X_train, Y_train_gender, Y_train_accent, X_test, Y_test_gender, Y_test_accent =  split_data(training_set)
    cPickle.dump([X_test, Y_test_gender, Y_test_accent], open('validation.pkl','wb'),-1)
    print 'Split data'
    print 'number of training set', len(X_train)
    print 'number of validation set', len(X_test)
    inputs = Input(shape=(maxlen,N_FEATURES,))
    #lstm1 = LSTM(16, return_sequences=True)(inputs)
    lstm1 = Bidirectional(LSTM(16, return_sequences=True)(inputs), merge_mode='concat')
    lstm1_flatten = Flatten()(lstm1)
    #lstm1 = LSTM(1024)(inputs)
    lstm1_drop = Dropout(0.5)(lstm1_flatten)
    dense1_gender = Dense(256, activation='relu')(lstm1_drop)
    dense2_gender = Dense(64, activation='relu')(dense1_gender)
    #dense2_gender_drop = Dropout(0.5)(dense2_gender)
    #dense3_gender = Dense(16, activation='relu')(dense2_gender_drop)
    output_gender = Dense(2, activation='softmax')(dense2_gender)
    
    dense1_accent = Dense(256, activation='relu')(lstm1_drop)
    dense2_accent = Dense(64, activation='relu')(dense1_accent)
    #dense2_accent_drop = Dropout(0.5)(dense2_accent)
    dense3_accent = Dense(16, activation='relu')(dense2_accent)
    output_accent = Dense(3, activation='softmax')(dense3_accent)
    
    model = Model(input=inputs, output=[output_gender, output_accent])
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    
    '''model'''
#    model = Sequential()
#    model.add(InputLayer(input_shape=(maxlen,13,)))
#    model.add(LSTM(64, return_sequences=True))
#    model.add(Flatten())
#    model.add(Dense(32))
#    model.add(Dense(2,activation='softmax'))
#    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, \
                              verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    #model.fit(X_train, Y_train_gender, batch_size=10, verbose=1, epochs=3)
    model_info = model.fit(X_train, [Y_train_gender, Y_train_accent], validation_data=(X_test, [Y_test_gender, Y_test_accent]), \
              batch_size=128, verbose=2, callbacks=callbacks_list, epochs=1000)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    print("Saved model to disk")
    
def test_model():
    X_test, Y_test_gender, Y_test_accent = cPickle.load(open('validation.pkl','rb'))
    print 'X_test', len(X_test)
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")
    # labels = loaded_model.predict(X_test)
    # gender_pred = np.argmax(labels[0], axis = 1)
    # accent_pred = np.argmax(labels[1], axis = 1)
    # print gender_pred
    # print accent_pred
    loaded_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    acc = loaded_model.evaluate(X_test, [Y_test_gender, Y_test_accent], batch_size=128)
    #print('Test score:', score)
    print('Test accuracy:', acc)
    
if __name__ == "__main__":
    training_set, maxlen, max_cepstrum, min_cepstrum = extract_features(path)
#    print 'number of training samples', len(training_set)
# #    print maxlen
# #    print max_cepstrum
# #    print min_cepstrum
#    training_set = normalize_data(training_set, min_cepstrum=min_cepstrum, max_cepstrum=max_cepstrum)
#    maxlen = get_new_max_length(training_set)
#    print 'new max length for padding', maxlen
#    training_set = get_same_length_data(training_set, maxlen=maxlen)
#    
## #    for data in training_set:
## #        assert data['feature'].shape[0] == maxlen
#    model(training_set, maxlen)
#    test_model()
#    #print training_set[0:2]['gender']

    
    























