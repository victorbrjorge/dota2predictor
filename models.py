#!/usr/bin/python
#-*- coding: utf-8 -*-
# encoding: utf-8

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
import time

def train_model(dataset):
    x = dataset.combine_id.values.tolist()
    y = dataset.radiant_win.values.tolist()

    for i in range(len(x)):
        x[i] = np.array(eval(x[i]))
    x = np.array(x)

    start = time.time()
    model_sxgb = XGBClassifier(nthread=8)
    model_xgb = XGBClassifier(nthread=8, n_estimators=200,learning_rate=0.3, max_depth=2, subsample=0.6, colsample_bytree=0.8, colsample_bylevel=0.4)
    model_adab = AdaBoostClassifier(n_estimators=1000, learning_rate=0.8)
    model_bb = BernoulliNB(binarize=None)
    
    results_sxgb = cross_val_score(model_sxgb, x, y, cv=5, n_jobs=-1)
    results_xgb = cross_val_score(model_xgb, x, y, cv=5, n_jobs=-1)
    results_adab = cross_val_score(model_adab, x, y, cv=5, n_jobs=-1)
    results_bb = cross_val_score(model_bb, x, y, cv=5, n_jobs=-1)
    elapsed = time.time() - start
    
    print("Time elapsed: %f" % (elapsed))
    print("S-XGBoost Accuracy: %.2f%% (%.2f%%)" % (results_sxgb.mean()*100, results_sxgb.std()*100))
    print("XGBoost Accuracy: %.2f%% (%.2f%%)" % (results_xgb.mean()*100, results_xgb.std()*100))
    print("AdaBoost Accuracy: %.2f%% (%.2f%%)" % (results_adab.mean()*100, results_adab.std()*100))
    print("NB Accuracy: %.2f%% (%.2f%%)" % (results_bb.mean()*100, results_bb.std()*100))

    y = np.array([results_sxgb.mean(), results_xgb.mean(), results_adab.mean(), results_bb.mean()])
    e = np.array([results_sxgb.std(), results_xgb.std(), results_adab.std(), results_bb.std()])

    np.savez(open('out', 'w'), y=y, e=e)

def grid_search(param_grid):
    
    kfold = StratifiedKFold(y, n_folds=5, shuffle=True)
    grid_search = GridSearchCV(model, param_grid, n_jobs=8, cv=kfold, verbose=10)
    
    result = grid_search.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means, stdevs = [], []
    for params, mean_score, scores in result.grid_scores_:
        stdev = scores.std()
        means.append(mean_score)
        stdevs.append(stdev)
        print("%f (%f) with: %r" % (mean_score, stdev, params))    



if __name__ == '__main__':
    train_model(pd.read_csv('dota_matches.csv'))