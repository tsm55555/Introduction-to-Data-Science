# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import math
import matplotlib.pyplot as plt
import numpy
from numpy.lib.function_base import gradient
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
# my import
import time

def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./HTRU_2.csv', header=None, names=['x%i' % (i) for i in range(8)] + ['y'])
    X = numpy.asarray(data[['x%i' % (i) for i in range(8)]])
    X = numpy.hstack((numpy.ones((X.shape[0],1)), X))
    y = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

def cross_entropy(y, y_hat):
    loss = 0
    for i in range(len(y)):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
        #print("loss ", i, ": ", loss)
    return loss


def logreg_sgd(X, y, alpha = .001, epochs = 10000, eps=1e-4):
    # TODO: compute theta
    # alpha: step size
    # epochs: max epochs
    # eps: stop when the thetas between two epochs are all less than eps 

    n, d = X.shape #  (8949, 9)
    #print("X shape: ", X.shape)
    #print("y.shape: ", y.shape)
    y = y.reshape(8949, 1)
    theta = numpy.zeros((d, 1))
    #print(theta)
    loss_previous = 1
    #formula: theta = theta - alpha * gradient
    for i in range(epochs):
        #print("epoch: ", i)
        y_hat = predict_prob(X, theta)
        #print("y_hat: ", y)
        loss = y_hat - y
        gradient = numpy.dot(X.T, loss) 
        theta -= alpha * gradient

        # calculate thetas between two epochs, stop if lower than eps(in this case it never stop lmao)
        cross_entropy_loss = cross_entropy(y, y_hat)
        if(abs(loss_previous - cross_entropy_loss) < eps):
            print("lower than eps break")
            break
        loss_previous = cross_entropy_loss

    return theta


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    tpr = []
    fpr = []
    #fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, y_prob)
    for thresholds in numpy.arange(0, 1, 0.001):
        #print("thresholds:", thresholds)
        TP = FP = FN = TN = 0
        for y_index, y in  enumerate(y_prob):
            if y > thresholds and y_test[y_index] == 1:
                TP += 1
            elif y < thresholds and y_test[y_index] == 1:
                FN += 1
            elif y > thresholds and y_test[y_index] == 0:
                FP += 1
            elif y < thresholds and y_test[y_index] ==0:
                TN += 1
        current_tpr = TP/(TP+FN)
        current_fpr = FP/(FP+TN)
        #print("current_tpr: ", current_tpr)
        #print("current_fpt: ", current_fpr)
        tpr.append(current_tpr)
        fpr.append(current_fpr)

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("roc_curve.png")


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    
    start_time = time.time()
    theta = logreg_sgd(X_train_scale, y_train)
    end_time = time.time()
    print("time elapsed:", end_time-start_time)

    print(theta)
    # theta = [[0.]
    #         ,[ -9.60770821]
    #         ,[  0.44053858]
    #         ,[ 16.9074481 ]
    #         ,[  0.80578103]
    #         ,[ -5.81422069]
    #         ,[  2.75235616]
    #         ,[-11.04834491]
    #         ,[  3.6384396 ]]
    y_prob = predict_prob(X_train_scale, theta)
    print("Logreg train accuracy: %f" % (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    print("Logreg train precision: %f" % (sklearn.metrics.precision_score(y_train, y_prob > .5)))
    print("Logreg train recall: %f" % (sklearn.metrics.recall_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" % (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    print("Logreg test precision: %f" % (sklearn.metrics.precision_score(y_test, y_prob > .5)))
    print("Logreg test recall: %f" % (sklearn.metrics.recall_score(y_test, y_prob > .5)))
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)


