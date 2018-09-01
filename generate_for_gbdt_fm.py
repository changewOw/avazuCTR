import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
import time
import utils_my
from utils_my import *
import sys
import os
from sklearn.utils import check_random_state
from sklearn.externals.joblib import dump, load


t0 = load("t0.joblib_dat")

print("t0 loaded with shape,",t0.shape)
#原始属性和appsiteid结合并转成category
#生成了_A_vn = add(app_site_id + vn)
vns0 = ['app_or_web', 'banner_pos', 'C1', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
for vn in vns0 + ['C14']:
    vn2 = '_A_' + vn
    print(vn2)
    t0[vn2] = np.add(t0['app_site_id'].values,t0[vn].astype('str').values)
    t0[vn2] = t0[vn2].astype('category')

t3 = t0
#raw + _A_vn + count/sequence + app&site
#lack as_domain as_category as_model
vns1 = vns0 + ['hour1'] + ['_A_' + vn for vn in vns0] +\
        ['device_model', 'device_type', 'device_conn_type', 'app_site_id', 'as_domain', 'as_category',
        'cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
        'cnt_diff_device_ip_day_pday','as_model','cnt_device_ip_day'] + \
        ['dev_id_cnt2', 'dev_ip_cnt2','C14','_A_C14','dev_ip2plus', 'dev_id2plus']

#cnt_device_ip_day
t3a = t3.ix[:,['click']].copy()
idx_base = 3000
for vn in vns1:
    if vn in ['cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
    'cnt_diff_device_ip_day_pday', 'cnt_device_ip_day']:
        #为什么要限制再-100和200之间
        #for i in ['cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
        #'cnt_diff_device_ip_day_pday', 'cnt_device_ip_day']:
            #filter1 = np.logical_and(t0[i].values>=-100,t0[i].values<=200)
            #print(t0.ix[filter1,'click'].shape[0],t0.shape[0])
            #4046334 4050240
            #4048443 4050240
            #4048404 4050240
            #3806941 4050240
            #3882547 4050240
            #3845574 4050240
            #-100 和200占的比例相当大，gbdt容易过拟合所以要去掉一些泛化能力差的数据，如diff存在2000的值
        _cat = pd.Series(np.maximum(-100, np.minimum(200, t3[vn].values))).astype('category').values.codes
    elif vn in ['as_domain']:
        _cat = pd.Series(np.add(t3['app_domain'].values,t3['site_domain'].values)).astype('category').values.codes
    elif vn in ['as_category']:
        _cat = pd.Series(np.add(t3['app_category'].values,t3['site_category'].values)).astype('category').values.codes
    elif vn in ['as_model']:
        _cat = pd.Series(np.add(t3['app_site_id'].values,t3['device_model'].values)).astype('category').values.codes
    else:
        _cat = t3[vn].astype('category').values.codes
    _cat = np.asarray(_cat, dtype='int32')
    _cat1 = _cat + idx_base
    t3a[vn] = _cat1
    print(vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size)
    #idx =idx +4500+1
    idx_base += _cat.max() + 1
#这里用idx是因为后续要用fm，参考fm的格式
print("to save t3a ....")
t3a_save = {}
t3a_save['t3a'] = t3a
t3a_save['idx_base'] = idx_base
dump(t3a_save,"t3a.joblib_dat")
"""
app_or_web 3000 3000 3001 2
banner_pos 3002 3002 3008 7
C1 3009 3009 3015 7
C15 3016 3016 3023 8
C16 3024 3024 3032 9
C17 3033 3033 3457 425
C18 3458 3458 3461 4
C19 3462 3462 3527 66
C20 3528 3528 3694 167
C21 3695 3695 3754 60
hour1 3755 3755 3778 24
_A_app_or_web 3779 3779 12102 8324
_A_banner_pos 12103 12103 20592 8490
_A_C1 20593 20593 28916 8324
_A_C15 28917 28917 39292 10376
_A_C16 39293 39293 48804 9512
_A_C17 48805 48805 106659 57855
_A_C18 106660 106660 123022 16363
_A_C19 123023 123023 156816 33794
_A_C20 156817 156817 203390 46574
_A_C21 203391 203391 238247 34857
device_model 238248 238248 244652 6405
device_type 244653 244653 244657 5
device_conn_type 244658 244658 244661 4
app_site_id 244662 244662 252985 8324
as_domain 252986 252986 257525 4540
as_category 257526 257526 257576 51
cnt_device_ip_day_hour 257577 257577 257752 176
cnt_device_ip_day_hour_prev 257753 257753 257929 177
cnt_device_ip_day_hour_next 257930 257930 258106 177
cnt_device_ip_pday 258107 258107 258301 195
cnt_diff_device_ip_day_pday 258302 258302 258544 243
as_model 258545 258545 502907 244363
cnt_device_ip_day 502908 502908 503100 193
dev_id_cnt2 503101 503101 503196 96
dev_ip_cnt2 503197 503197 503473 277
C14 503474 503474 505904 2431
_A_C14 505905 505905 621636 115732
dev_ip2plus 621637 621637 1069420 447784
dev_id2plus 1069421 1069421 1160465 91045
"""



