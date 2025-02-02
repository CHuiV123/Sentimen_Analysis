#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:52:55 2022

@author: angela
"""

import json
import os 
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model 


#%%

import tensorflow as tf

print(tf.__version__)

#%% Models Loading 
# Tokenizer 
TOKENIZER_PATH = os.path.join(os.getcwd(),'models','tokenizer.json')

with open(TOKENIZER_PATH,'r') as file: 
    loaded_tokenizer = json.load(file)


tokenizer = tokenizer_from_json(loaded_tokenizer)


# Model 


MODEL_PATH = os.path.join(os.getcwd(),'models','model.h5')

model = load_model(MODEL_PATH)

model.summary()

# OHE 
OHE_PATH = os.path.join(os.getcwd(),'models','ohe.pkl')

with open (OHE_PATH,'rb') as file: 
    ohe = pickle.load(file)

#%% 
# Data Loading 

new_review = ["Another Biggie from the Deadpool franchise hits hard! Goosebumping effects and stunts make it worth taking the day-off for. Amazing performance from the whole cast, everyone was just perfect in his/her space. Sense of humour was always perfect and r rated movies from marvel-20th century is unusual but it's like they have perfected that too. Every piece of the story was perfectly timelined and placed/explained. The conversations amongst wade and his wife could have been a little dramatic because it breaks the funny flow/vibe of the movie all along but the idea of saving the child for his own better future is beautiful. Even this is kind of satisfying and fairy-likewhn wade hits lows and meets his wif e for help like a pixie to a princess. Anyways, the movie does have some fairytale stuff unlike the first part but still is great."]
new_review = [input('Please type your data here ')]

import numpy as np
import re 
from tensorflow.keras.preprocessing.sequence import pad_sequences  

#data cleaning
for index, text in enumerate(new_review):  #loop it to do it for all data
    new_review = re.sub('<.*?>','',text)
    new_review = re.sub('[^a-zA-Z]',' ',new_review).lower().split()



#data preprocessing
new_review = tokenizer.texts_to_sequences(new_review)
new_review=np.reshape(new_review,(1,len(new_review)))
new_review = pad_sequences(new_review,maxlen=178,padding='post',truncating='post')


#%% model prediction 

outcome = model.predict(new_review)
print('This is review is {}'.format(ohe.inverse_transform(outcome)[0][0]))




