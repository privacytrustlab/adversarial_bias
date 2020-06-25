import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
from collections import namedtuple
from fairlearn.fairlearn.reductions import EqualizedOdds, EqualOpportunity
from fairlearn.fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.fairlearn.reductions import ExponentiatedGradient

def train_unconstraind_model(model, x_train, y_train, **kwargs):
    model.fit(x_train, y_train, **kwargs)
    return model.predict

def train_fair_model_reduction(base_model, x_train, y_train, g_train, fair_function, gap, sw=None, **kw):
    """
    Return the fairest classifier among the generated ones
    """
    fair_model = ExponentiatedGradient(base_model, fair_function, eps=gap)
    fair_model.fit(x_train, y_train, sensitive_features=g_train, sw=sw, **kw)
    return fair_model._best_classifier
def train_fair_model_post_processing(base_model, x_train, y_train, g_train, fair_function, **kw):
    fair_model = ThresholdOptimizer(estimator=base_model, constraints=fair_function)
    fair_model.fit(x_train, y_train, sensitive_features=g_train, **kw)
    pred_function = lambda x,g: fair_model._pmf_predict(x,sensitive_features=g)[:,1]
    return pred_function