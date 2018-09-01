import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
import pylab 
import sys
import time
import os
import utils
from utils import *
from sklearn.externals.joblib import dump, load, Parallel, delayed
import xgboost as xgb

rseed = 0
xgb_eta = .3
n_passes = 5
n_trees = 40
n_iter = 7
n_threads = 8
nr_factor = 4
learning_rate = .1
test_day = 30

def build_data():
    """
    what's the t0tv_mx:
    t0tv_mx is a matrix
    'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type'
    count features
    diffs features
    exptv features in both unvariate and multivariate
    """
    t0tv_mx_save = load("t0tv_mx.joblib_dat")
    #读入t0tv matrix
    t0tv_mx = t0tv_mx_save['t0tv_mx']
    click_values = t0tv_mx_save['click']
    day_values = t0tv_mx_save['day']
    print("t0tv_mx loaded with shape",t0tv_mx.shape)
    #均匀分布
    np.random.seed(rseed)
    nn = t0tv_mx.shape[0]
    r1 = np.random.uniform(0, 1, nn)

    #22->29的四分之一
    filter1 = np.logical_and(np.logical_and(day_values >= 22, day_values < test_day), np.logical_and(r1 < 0.25, True))
    #30
    filter_v1 = day_values == test_day
    #22->29均匀取四分之一
    xt1 = t0tv_mx[filter1, :]
    yt1 = click_values[filter1]

    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        print(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')
    #DMatrix train and test
    dtrain = xgb.DMatrix(xt1,label=yt1)
    dvalid = xgb.DMatrix(t0tv_mx[filter_v1],label=click_values[filter_v1])
    watchlist = [(dtrain,'train'),(dvalid,'valid')]
    print(xt1.shape, yt1.shape)

    param = {'max_depth':6,'eta':.5,'objective':'binary:logistic','verbose':0,
    'subsample':1.0, 'min_child_weight':50, 'gamma':0,
    'colsample_bytree':.5, 'base_score':0.16, 'seed': rseed,'eval_metric': 'logloss'}

    xgb_test_basis_d6 = xgb.train(param,dtrain,n_trees,watchlist)

    dtv = xgb.DMatrix(t0tv_mx)
    xgb_leaves = xgb_test_basis_d6.predict(dtv,pred_leaf=True)

    print(xgb_leaves.shape)
    #保存以便观察xgb_leaves
    dump(xgb_leaves,"xgb_leaves.joblib_dat")

    for i in range(n_trees):
