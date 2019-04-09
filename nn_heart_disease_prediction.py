# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 22:14:15 2019
@author: Paritosh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('heart.csv')

X = dataset.iloc[:, 0:13].values
Y = dataset.iloc[:, 13].values #target 0 or 1
#print(dataset)
print("Old X:",X)
print("Y:")
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
#NOW AS WE HAVE WIDE RANGE OF DATA WHERE SOME IS 1000'S AND SOME ARE ONLY SINGLE DIGIT OR 10'S DIGIT ETC
# TO GET THE BEST ACCURATE RESULTS WE WILL PERFORM SCALING ON OUR DATA SO ALL ARE EQUALLY CONTRIBUTING
# WE USE -> StandardScaler in same scikitLearn Library.
# Feature Scaling;

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print("X-train is:",X_train)
print("X-test :", X_test)

# WE NEED SEQUENTIAL MODULE FOR INITIALIZING NN AND DENSE_MODULE TO ADD HIDDEN LAYER
#IMPORTING KERAS
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing NEURAL NETWORK
classifier = Sequential()

# We will add hidden layers one by one using dense function.

"""Our first parameter is output_dim. It is simply the number of nodes you want to add to this layer.
#init is the initialization of Stochastic Gradient Decent. In Neural Network we need to assign weights
 to each mode which is nothing but importance of that node.

# At the time of initialization, weights should be close to 0 and we will randomly initialize weights using uniform function.
 input_dim parameter is needed only for first layer as model doesn’t know the number of our input variables

#Activation Function: Very important to understand. Neuron applies activation function to weighted sum(summation of Wi * Xi 
where w is weight, X is input variable and i is suffix of W and X). 
#The closer the activation function value to 1 the more activated is the neuron and more the neuron passes the signal. 

#Here we are using rectifier(relu) function in our hidden layer and Sigmoid function in our output layer as we want binary result 
from output layer but if the number of categories in output layer is more than 2 then use SoftMax function.
"""
# Adding the input layer
classifier.add(Dense(output_dim= 8, init = 'uniform', activation = 'relu', input_dim = 13))

#Adding the first hidden layer as
classifier.add(Dense(output_dim = 6, init = 'uniform' ,activation = 'relu'))

#Adding the second hidden layer as
classifier.add(Dense(output_dim = 5, init = 'uniform' ,activation = 'relu'))


#adding the last layer i.e output layer as sigmoid function
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

"""
First argument is Optimizer, this is nothing but the algorithm you wanna use to find optimal set of weights(Note that
in step 9 we just initialized weights now we are applying some sort of algorithm which will optimize weights in turn 
making out neural network more powerful. 
This algorithm is Stochastic Gradient descent(SGD). Among several types of SGD algorithm the one which we will use is ‘Adam’. If you go in deeper detail of SGD, you will find that SGD depends
on loss thus our second parameter is loss. Since out dependent variable is binary, we will have to  use logarithmic loss
function called ‘binary_crossentropy’, if our dependent variable has more than 2 categories in output then use 
‘categorical_crossentropy’. We want to improve performance of our neural network based on accuracy so add metrics as accuracy
"""

#Compiling the Neural Network
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

###
#using adagrad
classifier.compile(optimizer='adagrad',loss='binary_crossentropy', metrics = ['accuracy'])
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
## 88.52%
###

#using 
###

###
"""
We will now train our model on training data but still one thing is remaining. We use fit method to the fit our model 
In previous some steps I said that we will be optimizing our weights to improve model efficiency so when are we updating
 out weights? 
 Batch size is used to specify the number of observation after which you want to update weight. Epoch is 
 nothing but the total number of iterations. Choosing the value of batch size and epoch is trial and error there is no 
 specific rule for that.
"""

#Fitting our Model
classifier.fit(X_train, Y_train, batch_size = 13, nb_epoch = 300)

"""
Predicting the test set result. The prediction result will give you probability of Heart Attack.
 We will convert that probability into binary 0 and 1.
"""
# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

"""
This is the final step where we are evaluating our model performance. We already have original results and 
thus we can build confusion matrix to check the accuracy of model.
"""

#Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print("CM: ",cm)
ac=((cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1]))
print("Accuracy is: ",ac*100," %")