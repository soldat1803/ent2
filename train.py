#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import io
import nltk
import keras as k
from sklearn.metrics import f1_score,accuracy_score
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, merge, Dropout, RepeatVector, Activation, merge, Lambda, Flatten, Reshape
from keras.layers import LSTM, Bidirectional, TimeDistributed, GRU, concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adam, RMSprop , Adagrad ,Adamax,Nadam,SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import string
import argparse
from collections import Counter
import re
import argparse
import json
import sys
# from utils copy preprocess.py and attention.py to colab local location
from preprocess import splitDatasets, splitValDatasets,vectorizeValData, vectorizeData
from attention import Attention


# In[2]:


embeddings_index = {}
f = open('glove/glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[3]:


with open('data/train-v2.0.json') as json_data:
    d = json.load(json_data)
trainData = d['data']


# In[4]:


tContext, tQuestion, tQuestion_id, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = splitDatasets(trainData)


# In[5]:



with open('data/dev-v2.0.json') as json_data:
    d = json.load(json_data)
valData = d['data']


# In[6]:


vContext, vQuestion, vQuestion_id, vAnswerBegin, vAnswerEnd, vAnswerText, vmaxLenTContext, vmaxLenTQuestion = splitDatasets(valData)


# In[7]:


vocab = {}
for words in tContext + tQuestion + vContext + vQuestion:
    for word in words:
        if word not in vocab:
            vocab[word] = 1
vocab = sorted(vocab.keys())


# In[8]:


vocab_size = len(vocab) + 1
print(vocab_size)
word_index = dict((c, i + 1) for i, c in enumerate(vocab))
context_maxlen = max(maxLenTContext, vmaxLenTContext)
question_maxlen = max(maxLenTQuestion, vmaxLenTQuestion)


# In[9]:


tX, tXq, tYBegin, tYEnd = vectorizeData(tContext, tQuestion, tAnswerBegin, tAnswerEnd, word_index, context_maxlen, question_maxlen)


# In[10]:


vX, vXq, vYBegin, vYEnd = vectorizeData(vContext, vQuestion, vAnswerBegin, vAnswerEnd, word_index, context_maxlen, question_maxlen)


# In[11]:


nb_words = len(word_index)
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = context_maxlen

embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[51]:


def train(units,opt,lr):
    question_input = Input(shape=(question_maxlen,), dtype='int32', name='question_input')
    context_input = Input(shape=(context_maxlen,), dtype='int32', name='context_input')
    questionEmbd = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,mask_zero=True, weights=[embedding_matrix], 
                         input_length=question_maxlen, trainable=False)(question_input)
    contextEmbd = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,mask_zero=True, weights=[embedding_matrix], 
                input_length=context_maxlen, trainable=False)(context_input)
    Q = Bidirectional(LSTM(units, return_sequences=True))(questionEmbd)
    D = Bidirectional(LSTM(units, return_sequences=True))(contextEmbd)
    Q2 = Attention(question_maxlen)(Q)
    D2 = Attention(context_maxlen)(D)
    L = concatenate([D2, Q2], axis=1)
    answerPtrBegin_output = Dense(context_maxlen, activation='softmax')(L)
    Lmerge = concatenate([L, answerPtrBegin_output],axis = 1)
    answerPtrEnd_output = Dense(context_maxlen, activation='softmax')(Lmerge)
    model = Model(input=[context_input, question_input], output=[answerPtrBegin_output, answerPtrEnd_output])
    am = opt(lr=0.0002)
    model.compile(optimizer=am, loss='categorical_crossentropy',loss_weights=[.04, 0.04], metrics=['accuracy'])
    model.summary()
    return model


# In[52]:



train_slice = 10000
filepath="modelweightsfinal.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[53]:


epoch = [10,20,30,40,50,80,100]
units = [32,64,128,256,512]
lr=[0.0002,0.0003,0.0004,0.0008,0.001,0.002,0.003,0.006,0.01,0.02,0.03,0.04,0.05]
list_opt=[optimizers.Adam,optimizers.RMSprop,optimizers.Adagrad,optimizers.Adamax,optimizers.Nadam,optimizers.SGD]


# In[54]:


for e in epoch:
    for u in units:
        for lst in list_opt:
            for learn in lr:
            
                model = train(u,lst,lr)
                model_history = model.fit([tX, tXq], [tYBegin, tYEnd],batch_size= 128, verbose=1,
                          callbacks = callbacks_list,epochs=30)


# In[14]:





# In[15]:





# In[16]:





# In[17]:





# In[18]:





# In[19]:





# In[ ]:





# In[ ]:




