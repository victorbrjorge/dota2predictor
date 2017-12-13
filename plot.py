#!/usr/bin/python
#-*- coding: utf-8 -*-
# encoding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_acc(y, e):
    labels = ['ST-XGboost', 'GS-XGBoost', 'AdaBoost', 'Naive Bayes']

    fig, ax = plt.subplots()
    ax.bar(range(len(y)), y, yerr=e, tick_label=labels, ecolor='red', color=['yellow', 'blue', 'green', 'orange'], align='center')

    ax.set_ylim([0.58,0.61])
    #ax.yaxis.grid()
    for v in y:
    	plt.axhline(y=v, ls='--', lw=0.5, color='black')
    
    plt.title(u'5-fold Cross Validation')
    plt.ylabel(u"Acurácia")
    #plt.xlabel(u"Algoritmo")
    plt.savefig('accuracy.png')

if __name__ == '__main__':
	file = np.load(open('out'))
	plot_acc(file['y'], file['e'])