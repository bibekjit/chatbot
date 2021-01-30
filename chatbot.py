import json
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem import LancasterStemmer

lanstem = LancasterStemmer()

intents = json.loads(open('intents.json').read())
# this file contains the message data for the chatbot

classes=[]
words=[]
docs=[]
ignore=[]

import string
for i in string.punctuation:
    ignore.append(i)
# appended all the punctuations in 'ignore'

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        docs.append([w,intent['tag']])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# tokenised the sentences and extended the tokens in 'words'
# appended 'tag' in 'classes' if it not present

words=[lanstem.stem(w.lower()) for w in words if w not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
# stemmed all the words of 'words' list and converted it into a set
# so as to remove duplicate words and sorted it

train=[]
out_emp=[0]*len(classes)

for doc in docs:
    pattern_words=[lanstem.stem(w.lower()) for w in doc[0]]
    bag=[]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    # stemmed the tokens 'docs'
    # created a bag and appended 1 if word in pattern_words
    # else appended 0

    out_row=list(out_emp)
    out_row[classes.index(doc[1])]=1
    train.append([bag,out_row])
    # initiated 1 for current tag in out_row
    # appended both bag and out_row to 'train'
    # to create the training data


import random
random.shuffle(train) # randomised the arrangements of values in 'train'
train=np.array(train,dtype='object')
x_train=list(train[:,0])
y_train=list(train[:,1])
# splitted the training data
# x_train -> stemmed tokens
# y_train -> tags

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
model=Sequential()
model.add(Dense(128,input_shape=(len(x_train[0]),),activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(y_train[0]),activation='softmax'))
# created a sequential model

model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
model.fit(np.array(x_train),np.array(y_train),epochs=200,batch_size=10)
# note -  the splitted data is converted into an array before fitting them

# import pandas as pd
# import matplotlib.pyplot as plt
# pd.DataFrame(model.history.history).plot()
# # analysed the 'loss' and 'accuracy' curve

def stemming(sent):
    """
    :param sent: message sentence
    :return: list of stemmed tokens
    """
    sentwords=nltk.word_tokenize(sent)
    sentwords=[lanstem.stem(w.lower()) for w in sentwords]
    return sentwords


def bow(sent,words):
    """
    :param sent: message sentence
    :param words: list of all the stemmed tokens available in the json file
    :return: bag of words in array form
    """
    sentwords=stemming(sent)
    bag=[0]*len(words)
    for s in sentwords:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1
    return np.array(bag)


def pred_class(sent,model):
    """
    :param sent: message sentence
    :param model: model used for creating the neural network
    :return: list of tags and its predicted probability
    """
    p=bow(sent,words)
    results=model.predict(np.array([p]))[0]
    results=[[i,r] for i,r in enumerate(results) if r>0.25]
    results.sort(key=lambda x: x[1],reverse=True)
    result_list=[]
    for r in results:
        result_list.append({'intent':classes[r[0]],'probability':r[1]})
    return result_list

def get_resp(ints,jsonfile):
    """
    :param ints: a list of tags with their predicted probablitites
    :param jsonfile: the intents.json file
    :return: response message by the chatbot
    """
    tag=ints[0]['intent']
    intent_list=jsonfile['intents']
    for i in intent_list:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result

def message(msg):
    """
    :param msg: message given by user
    :return: reply to the user's message
    """
    ints=pred_class(msg,model)
    reply=get_resp(ints,intents)
    return reply

for i in range(5):
    print('.')
print()
while True:
    your_text=input('you : ')
    print()
    print('bot :',message(your_text),'\n')
    if 'bye' in your_text:
        break

