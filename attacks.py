from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from numpy import linalg
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
import keras.backend as K
from keras.optimizers import Adam, RMSprop
from time import time
import pandas as pd
import os
from utils import *


class MyModel:
    """
    keras implementation of logistic regression
    """
    def __init__(self, input_shape, epochs):
        x = Input(shape=input_shape)
        y = Dense(1, activation='sigmoid')(x)
        y = Reshape((1,))(y)
        model = Model([x], y)
        model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
        self.learner = model
        self.epochs = epochs

    def fit(self, X, Y, sample_weight=None, batch_size=1000, verbose=0):
        self.learner.fit(X,Y,sample_weight=sample_weight, batch_size=batch_size, 
                         epochs=self.epochs, verbose=verbose)
    def score(self, X, Y, sample_weight=None):
        return self.learner.evaluate(X,Y,sample_weight=sample_weight)
    def predict(self, X):
        pred = self.learner.predict(X).flatten()
        return 1.0*(pred>0.5)
    def predict_proba(self, X):
        return self.learner.predict(X)

def uniform_sampling(datatrain, attacker_set, epsilon, flip=False):
    """
    datatrain: clean training data
    attacker_set: attacker dataset
    epsilon: fraction of poisoning data
    flip: False -> sampling (random sampling/hard examples) or True --> label flipping 
    """
    
    X_TRAIN = datatrain['x_train']
    Y_TRAIN = datatrain['y_train']
    G_TRAIN = datatrain['g_train']
    
    X_ATTACK = attacker_set['x_train']
    Y_ATTACK = attacker_set['y_train']
    G_ATTACK = attacker_set['g_train']
    
    if flip:
        Y_ATTACK = 1-Y_ATTACK

    
    num_points = int(epsilon * len(X_TRAIN))
    
    idx = np.random.choice(np.arange(len(X_ATTACK)), num_points, replace=False)

    X_TRAIN_p = X_ATTACK[idx]
    Y_TRAIN_p = Y_ATTACK[idx]
    G_TRAIN_p = G_ATTACK[idx]
    
    X_TRAIN_n = np.concatenate([X_TRAIN, X_TRAIN_p], axis=0)
    Y_TRAIN_n = np.append(Y_TRAIN, Y_TRAIN_p)
    G_TRAIN_n = np.append(G_TRAIN, G_TRAIN_p)
    
    return X_TRAIN_n, Y_TRAIN_n, G_TRAIN_n


def algorithm1(datatrain, attacker_set, epsilon, L, num_iters, r, lr, flip=True):
    """
    datatrain: clean training data
    attacker_set: attacker dataset
    epsilon: fraction of poisoning data
    L: lambda/epsilon (in the paper), where lambda is the penalty coefficient
    num_iters: T 
    r: L2 parameter regularization for the model
    lr: learning rate
    flip: False -> sampling (random sampling/hard examples) or True --> labeling
    """
    X_TRAIN = datatrain['x_train']
    Y_TRAIN = datatrain['y_train'].reshape((-1,1))
    G_TRAIN = datatrain['g_train'].reshape((-1,1))
    
    
    X_ATTACK = attacker_set['x_train']
    Y_ATTACK = attacker_set['y_train'].reshape((-1,1))
    G_ATTACK = attacker_set['g_train'].reshape((-1,1))
    
    if flip:
        X_ATTACK = np.concatenate([X_ATTACK, X_ATTACK], axis=0)
        Y_ATTACK = np.concatenate([Y_ATTACK, 1-Y_ATTACK], axis=0)
        G_ATTACK = np.concatenate([G_ATTACK, G_ATTACK], axis=0)
        

    
    num_points = int(epsilon * len(X_TRAIN))
    
    num_feats = X_TRAIN.shape[1]

    A = np.random.random((X_TRAIN.shape[1],1))
    b = np.random.random()
    X_points = []
    Y_points = []
    G_points = []
    selected_idx = []
    n = len(X_TRAIN)
    for j in range(num_iters):
        
        if len(selected_idx) >= num_points:
            selected_idx = selected_idx[-num_points+1:]
        
        loss = eval_loss(A, b, X_ATTACK, Y_ATTACK, G_ATTACK, X_TRAIN, Y_TRAIN, G_TRAIN, L, num_points).flatten()
        
        s_loss = sorted(loss, key=lambda x: -x)
        idx_i = 0
        for i in range(len(s_loss)):
            c = False
            for k in np.arange(len(loss))[loss==s_loss[i]]:
                if k not in selected_idx:
                    idx_i = k
                    c = True
                    break
            if c:
                break
        idx = idx_i
        selected_idx.append(idx)
        
        
        X_points.append(X_ATTACK[idx])
        Y_points.append(Y_ATTACK[idx])
        G_points.append(G_ATTACK[idx])
        
        
        
        x = X_ATTACK[idx].reshape((-1,num_feats))
        y = Y_ATTACK[idx].reshape((-1,1))
        g = G_ATTACK[idx].reshape((-1,1))
        
        x_c_copy = np.concatenate([X_TRAIN, np.repeat(x, num_points, axis=0)], axis=0)
        y_c_copy = np.concatenate([Y_TRAIN, np.repeat(y, num_points, axis=0)], axis=0)
        g_c_copy = np.concatenate([G_TRAIN, np.repeat(g, num_points, axis=0)], axis=0)
        

        for k in range(int(1)):
            gr = gradient(A, b, x_c_copy, y_c_copy, x_c_copy, y_c_copy, g_c_copy, L, r, n, num_points)
            A -= gr['dA'] * lr
            b -= gr['db'] * lr

    
    X_TRAIN_p = np.array(X_points).reshape((-1,num_feats))[-num_points:]
    Y_TRAIN_p = np.array(Y_points).reshape((-1,1))[-num_points:]
    G_TRAIN_p = np.array(G_points).reshape((-1,1))[-num_points:]

    X_TRAIN_n = np.concatenate([X_TRAIN, X_TRAIN_p], axis=0)
    Y_TRAIN_n = np.concatenate([Y_TRAIN, Y_TRAIN_p], axis=0).flatten()
    G_TRAIN_n = np.concatenate([G_TRAIN, G_TRAIN_p], axis=0).flatten()
    
    return X_TRAIN_n, Y_TRAIN_n, G_TRAIN_n


def algorithm2(datatrain, attacker_set, epsilon, L, num_iters, flip=True):
    """
    datatrain: clean training data
    attacker_set: attacker dataset
    epsilon: fraction of poisoning data
    L: lambda/epsilon (in the paper), where lambda is the penalty coefficient
    num_iters: T 
    flip: False -> sampling (random sampling/hard examples) or True --> labeling
    """

    X_TRAIN = datatrain['x_train']
    Y_TRAIN = datatrain['y_train'].reshape((-1,1))
    G_TRAIN = datatrain['g_train'].reshape((-1,1))
    
    
    X_ATTACK = attacker_set['x_train']
    Y_ATTACK = attacker_set['y_train'].reshape((-1,1))
    G_ATTACK = attacker_set['g_train'].reshape((-1,1))
    
    if flip:
        X_ATTACK = np.concatenate([X_ATTACK, X_ATTACK], axis=0)
        Y_ATTACK = np.concatenate([Y_ATTACK, 1-Y_ATTACK], axis=0)
        G_ATTACK = np.concatenate([G_ATTACK, G_ATTACK], axis=0)
        

    
    num_points = int(epsilon * len(X_TRAIN))
    
    num_feats = X_TRAIN.shape[1]
    
    model = MyModel((num_feats,), 1)
    model.fit(X_TRAIN, Y_TRAIN) # initialize parameters
    
    weights = np.ones(len(X_TRAIN))
    weights_ep = np.append(weights, num_points)

    X_points = []
    Y_points = []
    G_points = []
    selected_idx = []
    
    for j in range(num_iters):
        pred_proba = model.predict_proba(X_ATTACK)
        loss = cross_entropy(pred_proba, Y_ATTACK)
        
        pred = model.predict(X_ATTACK)
        pred_train = model.predict(X_TRAIN)
        
        EO_gap_hash = np.zeros(8) #hash table for EO gap
        for g_v in [0,1]:
            for y_v in [0,1]:
                for p_v in [0,1]:
                    yi = np.repeat(np.array(y_v, copy=True).reshape((1,-1)),num_points, axis=0)
                    gi = np.repeat(np.array(g_v, copy=True).reshape((1,-1)),num_points, axis=0)
                    pred_i = np.repeat(p_v, num_points)
                    Y_copy = np.concatenate([Y_TRAIN, yi], axis=0)
                    G_copy = np.concatenate([G_TRAIN, gi], axis=0)
                    pred_copy = np.append(pred_train, pred_i)
                    EO_gap_i = max(EO(G_copy, pred_copy, Y_copy))
                    EO_gap_hash[int(g_v*4+y_v*2+p_v)] = EO_gap_i
                    
        EO_gap = [EO_gap_hash[int(G_ATTACK[i]*4+Y_ATTACK[i]*2+pred[i])] for i in range(len(X_ATTACK))]
        EO_gap = np.array(EO_gap)

        loss = loss + L * EO_gap
        
        s_loss = sorted(loss, key=lambda x: -x)
        idx_i = 0
        for i in range(len(s_loss)):
            c = False
            for k in np.arange(len(loss))[loss==s_loss[i]]:
                if k not in selected_idx:
                    idx_i = k
                    c = True
                    break
            if c:
                break
                
        idx = idx_i
        selected_idx.append(idx)
        
        if len(selected_idx) >= num_points:
            selected_idx = selected_idx[-num_points:]


        x = np.array(X_ATTACK[idx], copy=True).reshape((1,-1))
        y = np.array(Y_ATTACK[idx], copy=True).reshape((1,-1))
        g = np.array(G_ATTACK[idx], copy=True).reshape((1,-1))

        X_points.append(x)
        Y_points.append(y)
        G_points.append(g)

        X_TRAIN_copy = np.concatenate([X_TRAIN, x], axis=0)
        Y_TRAIN_copy = np.concatenate([Y_TRAIN, y], axis=0)
        model.fit(X_TRAIN_copy, Y_TRAIN_copy, batch_size=1000, sample_weight=weights_ep, verbose=(j%100==0))
        
    X_TRAIN_p = np.array(X_points).reshape((-1,num_feats))[-num_points:]
    Y_TRAIN_p = np.array(Y_points).reshape((-1,1))[-num_points:]
    G_TRAIN_p = np.array(G_points).reshape((-1,1))[-num_points:]

    X_TRAIN_n = np.concatenate([X_TRAIN, X_TRAIN_p], axis=0)
    Y_TRAIN_n = np.concatenate([Y_TRAIN, Y_TRAIN_p], axis=0).flatten()
    G_TRAIN_n = np.concatenate([G_TRAIN, G_TRAIN_p], axis=0).flatten()
    
    return X_TRAIN_n, Y_TRAIN_n, G_TRAIN_n