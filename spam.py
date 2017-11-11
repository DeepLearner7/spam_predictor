#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:26:16 2017

@author: saurabh
"""
import numpy as np
import re
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def upload_glove():
    gloVe = open('glove.6B.50d.txt','r')
    wordsList = []
    wordVectors = []
    for line in gloVe:
        splitted_text = line.split()
        wordsList.append(splitted_text[0])
        wordVectors.append(splitted_text[1:])
        
    wordVectors = np.array(wordVectors)
    print('Number of words in gloVe model: ', len(wordVectors))
    
    return wordVectors, wordsList



def upload_csvFile():
    data = pd.read_csv("spam.csv", encoding='ISO-8859-1')
    
    info = data[data.Label=='info']
    info_len = len(info)
    modifiedIndex = [i for i in range(info_len)]
    info.index = modifiedIndex
    
    ham = data[data.Label=='ham']
    ham_len = len(ham)
    modifiedIndex = [i for i in range(info_len, info_len+ham_len)]
    ham.index = modifiedIndex
    
    spam = data[data.Label=='spam']
    spam_len = len(spam)
    modifiedIndex = [i for i in range(info_len+ham_len, info_len+ham_len+spam_len)]
    spam.index = modifiedIndex
    
    data = pd.DataFrame()
    data = data.append(info)
    data = data.append(ham)
    data = data.append(spam)
    
    return data, info_len, ham_len, spam_len



def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def createTensor(data, info_len, ham_len, spam_len, wordsList):
    numWords = []
    for msg in data.Message:
        count = len(msg.split())
        numWords.append(count)
        
    print("Total number of messages: ", len(data))
    print("Total number of words: ", sum(numWords))
    print("Average number of words per line: ", sum(numWords)/len(data))
    
    maxSeqLength = 25
    
    
    
    numMsg = len(data)
    
    try:
        inputTensor = np.load('preBuilt.npy')
    except:
        inputTensor = np.zeros((numMsg, maxSeqLength))
        for index in range(info_len):
            line = data.Message[index]
            cleanedLine = cleanSentences(line)
            splittedWord = cleanedLine.split()
            indexCounter = 0
            print(index)
            for word in splittedWord:
                try:
                    inputTensor[index][indexCounter] = wordsList.index(word)
                except:
                    inputTensor[index][indexCounter] = 399999
                indexCounter += 1
                if indexCounter >= maxSeqLength:
                    break
            
        
        for index in range(info_len, info_len+ham_len):
            line = data.Message[index]
            cleanedLine = cleanSentences(line)
            splittedWord = cleanedLine.split()
            indexCounter = 0
            print(index)
            for word in splittedWord:
                try:
                    inputTensor[index][indexCounter] = wordsList.index(word)
                except:
                    inputTensor[index][indexCounter] = 399999
                indexCounter += 1
                if indexCounter >= maxSeqLength:
                    break
            
        
        for index in range(info_len+ham_len, info_len+ham_len+spam_len):
            line = data.Message[index]
            cleanedLine = cleanSentences(line)
            splittedWord = cleanedLine.split()
            indexCounter = 0
            print(index)
            for word in splittedWord:
                try:
                    inputTensor[index][indexCounter] = wordsList.index(word)
                except:
                    inputTensor[index][indexCounter] = 399999
                indexCounter += 1
                if indexCounter >= maxSeqLength:
                    break
                
            
        np.save('preBuilt.npy', inputTensor)
        
    return inputTensor

def create_embedding_lookup(wordVectors, inputTensor):
    try:
        feedable_tensor = np.load('feedable.npy')
    except:
        
        feedable_tensor = np.zeros((30000, 25, 50))
        
        for i in range(inputTensor.shape[0]):
            for j in range(inputTensor.shape[1]):
                feedable_tensor[i][j] = wordVectors[inputTensor[i][j]]
                
        np.save('feedable.npy', feedable_tensor)
    return feedable_tensor

def output(info_len, ham_len, spam_len):
    encoded = np.zeros((30000, 3))
    encoded[:info_len] = [1, 0, 0]
    encoded[info_len:info_len+ham_len] = [0, 1, 0]
    encoded[info_len+ham_len:info_len+ham_len+spam_len] = [0, 0, 1]
    encoded = encoded.astype(int)
    return encoded


def buildModel(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(LSTM(128, input_shape=(25,50)))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train,  epochs=10, batch_size=64, verbose=1, validation_data=(x_test, y_test))
    output = model.predict(x_test)
    return output
    
if __name__ == "__main__":
    
    wordVectors, wordsList = upload_glove()
    data, info_len, ham_len, spam_len = upload_csvFile()
    
    
    inputTensor = createTensor(data, info_len, ham_len, spam_len, wordsList)
    
   
    feedable_tensor = create_embedding_lookup(wordVectors, inputTensor)
    
    
    y_one_hot_encoded = output(info_len, ham_len, spam_len)
    
    #X_train    
    X_train = feedable_tensor[:int(info_len*0.9)]
    X_train = np.append(X_train, feedable_tensor[info_len:info_len+int(ham_len*.9)], axis=0)
    X_train = np.append(X_train, feedable_tensor[info_len+ham_len:(info_len+ham_len+int(.9*spam_len))], axis=0)
    
    #Y_train
    Y_train = y_one_hot_encoded[:int(info_len*0.9)]
    Y_train = np.append(Y_train, y_one_hot_encoded[info_len:info_len+int(ham_len*.9)], axis=0)
    Y_train = np.append(Y_train, y_one_hot_encoded[info_len+ham_len:(info_len+ham_len+int(.9*spam_len))], axis=0)
    
    #X_test
    X_test = feedable_tensor[int(info_len*.9):info_len]
    X_test = np.append(X_test, feedable_tensor[info_len+int(ham_len*.9): info_len+ham_len], axis=0)
    X_test = np.append(X_test, feedable_tensor[info_len+ham_len+int(.9*spam_len): info_len+ham_len+spam_len], axis=0)
    

    #Y_test
    Y_test = y_one_hot_encoded[int(info_len*.9):info_len]
    Y_test = np.append(Y_test, y_one_hot_encoded[info_len+int(ham_len*.9): info_len+ham_len], axis=0)
    Y_test = np.append(Y_test, y_one_hot_encoded[info_len+ham_len+int(.9*spam_len): info_len+ham_len+spam_len], axis=0)
    
    #buildModel(feedable_tensor, y_one_hot_encoded)
    labeled_output = []
    output = buildModel(X_train, Y_train, X_test, Y_test)
    for i in range(output.shape[0]):
        index = np.argmax(output[i])
        if index == 0 :
            labeled_output.append('info')
        elif index == 1 :
            labeled_output.append('ham')
        else:
            labeled_output.append('spam')
        
    labeled_output = np.array(labeled_output)
    np.save("labeled_output", labeled_output)
    

    



 
    


