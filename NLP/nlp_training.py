#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:14:45 2022

@author: angela
"""


import pandas as pd 
import re
import os 
#%% 1) Data Loading 

df = pd.read_csv('https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv')

print('Hello World! Hello Warren!')

#%% 2) Data Inspection

df.head()
df.duplicated().sum()
df.isna().sum() #normally nothing much to check 

print(df['review'][4]) #<br/><br is html tags that we will want to remove

# Symbols and HTML tags have to be remove 


#%% 3) Data cleaning 

#test = df['review'][4]
#test.replace('br/>','') # this cant be use because we will need something robust. 
#there are too many types of html tags

# test = df['review'][4]
# print('__________BEFORE_______')
# print(test)
# print('__________AFTER_______')


# test = re.sub('<.*?>','',test)
# test = re.sub('[^a-zA-Z]',' ',test).lower().split()  #^ means except /not here, split is to make them in list form
# print(test)

review = df['review']
sentiment = df['sentiment']

review_backup = review.copy()
sentiment_backup = sentiment.copy()

for index, text in enumerate(review):
    # to remove html tags
    # Anything within the <> will be removed including <> 
    #? to tell re dont be greedy so it wont capture everything 
    # from the first < to the last > in the document 
    review[index] = re.sub('<.*?>','',text)
    review[index] = re.sub('[^a-zA-Z]',' ',text).lower().split()


#%% 4) Features Selection 

# there is no feature to select here

#%% 5) Data preprocessing 

# at this stage, we need to make sure there is no any empty space is our data
# use tokenizer, one hot encoding

from tensorflow.keras.preprocessing.text import Tokenizer 

vocab_size = 10000 # 1/5 of data size 
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review)  # to learn 
word_index = tokenizer.word_index

print(dict(list(word_index.items())[0:10]))  # to slice the data, show only 10
print(dict(list(word_index.items())))   # to show all 

review_int = tokenizer.texts_to_sequences(review) # To convert into numbers 
review_int[100]


length_review = []
for i in range(len(review_int)): 
    length_review.append(len(review_int[i]))
    # print(len(review_int[i]))

import numpy as np 
np.median(length_review)

max_len = np.median([len(review_int[i])for i in range(len(review_int))])

from tensorflow.keras.preprocessing.sequence import pad_sequences 

padded_review=pad_sequences(review_int,
                            maxlen=int(max_len),
                            padding='post',
                            truncating='post')


#Y target
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse=False)
sentiment=ohe.fit_transform(np.expand_dims(sentiment,axis=-1))


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(padded_review,sentiment,
                                               test_size=0.3,
                                               random_state=(123))


#%% model development 

from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional 
from tensorflow.keras import Input, Sequential
from tensorflow.keras.utils import plot_model 

# X_train=np.expand_dims(X_train,axis=-1)  
# X_test=np.expand_dims(X_test,axis=-1)
#> when u use embedding, u do not need to expand dim anymore. input shape is no longer important

input_shape = np.shape(X_train)[1:]
out_dim = 128 # u can try 64, 128 is just to match the rest of the layer here 

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(Embedding(vocab_size,out_dim))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.summary()


plot_model(model, show_shapes=(True))

model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics=['acc'])


#%% 
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping 
import datetime


LOGS_PATH = os.path.join(os.getcwd(),
                         'logs',
                         datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

tensorboard_callback=TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)
# early_callback = EarlyStopping(monitor = 'val_loss', patience=5)


hist = model.fit(X_train, y_train, 
                 epochs=5,
                 validation_data=(X_test, y_test),
                 callbacks=[tensorboard_callback])

#%% model analysis 

from sklearn.metrics import classification_report 

y_pred = np.argmax(model.predict(X_test),axis=1)
y_actual = np.argmax(y_test, axis=1)

print(classification_report(y_actual,y_pred))


#%% model saving 

# TOKENIZER 
import json 

TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'models','tokenizer.json')

token_json = tokenizer.to_json()

with open(TOKENIZER_SAVE_PATH,'w') as file: 
    json.dump(token_json,file)



# OHE 
import pickle

OHE_SAVE_PATH = os.path.join(os.getcwd(),'models','ohe.pkl')

with open(OHE_SAVE_PATH,'wb') as file: 
    pickle.dump(ohe,file)



# MODEL

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'models','model.h5') #model here refer to the mdeol = Sequential()
model.save(MODEL_SAVE_PATH)
