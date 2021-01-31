import json
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem import LancasterStemmer

lanstem = LancasterStemmer()

# this file contains the message data for the chatbot
intents = json.loads(open('intents.json').read())

classes=[]
words=[]
docs=[]
ignore=[]

# appending all the punctuations in 'ignore'
import string
for i in string.punctuation:
    ignore.append(i)

# tokenising the sentences and extending the tokens in 'words'
# appending 'tag' in 'classes' if it not present
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        docs.append([w,intent['tag']])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stemming all the words of 'words' list and converting it into a set
# so as to remove duplicate words and then sorting it
words=[lanstem.stem(w.lower()) for w in words if w not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))


train=[] # training dataset
out_emp=[0]*len(classes)


for doc in docs:
    # stemming the tokens in 'docs'
    # creating a bag of words and appending 1 if word in pattern_words
    # else appending 0
    pattern_words=[lanstem.stem(w.lower()) for w in doc[0]]
    bag=[]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
   
    # initiating 1 for current tag in out_row
    # appending both bag and out_row to 'train'
    # to create the training data
    out_row=list(out_emp)
    out_row[classes.index(doc[1])]=1
    train.append([bag,out_row])
    


import random
random.shuffle(train) # randomising the arrangements of values in 'train'

# splitted the training data
train=np.array(train,dtype='object')
x_train=list(train[:,0]) # stemmed tokens
y_train=list(train[:,1]) # tags

# creating a sequential model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
model=Sequential()
model.add(Dense(128,input_shape=(len(x_train[0]),),activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(y_train[0]),activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
model.fit(np.array(x_train),np.array(y_train),epochs=200,batch_size=10)
# note -  the splitted data is converted into an array before fitting them

# # illustrating the 'loss' and 'accuracy' curve
# import pandas as pd
# import matplotlib.pyplot as plt
# pd.DataFrame(model.history.history).plot()


def stemming(sent):
    """
    tokenises a sentence and then stems the tokens

    :param sent: message sentence (string)
    :return: list of stemmed tokens
    """
    sentwords=nltk.word_tokenize(sent)
    sentwords=[lanstem.stem(w.lower()) for w in sentwords]
    return sentwords


def bow(sent,words):
    """
    Creates a list of zeros. Looks for all the words present in 'sentwords'
    in the 'words' list and replaces 0 with 1 in the bag for the corresponding
    index position of the word in the 'word' list present in 'sentwords'

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
    Predicts the probability and tag of each token.
    Probabilities greater than 0.25 are listed and sorted.
    Each tag and probability is written in dictionary form
    and appended into a list

    :param sent: message sentence
    :param model: model used for creating the neural network
    :return: list of predicted tags and probabilities
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
    Selects the very first tag (tag with the highest probability)
    and returns a response message with respect to that tag

    :param ints: a list of tags with their predicted probablitites
    :param jsonfile: the intents.json file
    :return: response message by the chatbot (string)
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
    returns a response in string form by the chatbot

    :param msg: message received by the chatbot
    :return: reply to the user's message
    """
    ints=pred_class(msg,model)
    reply=get_resp(ints,intents)
    return reply

for i in range(5):
    print('.')
print()

# interacting with the chatbot
while True:
    your_text=input('you : ')
    print()
    print('bot :',message(your_text),'\n')
    if 'bye' in your_text:
        break

