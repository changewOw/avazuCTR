import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
import time
from sklearn.externals.joblib import dump, load, Parallel, delayed

def calcDualKey(df,vn,vn2,key_src,key_tgt,vn_y,cred_k,mean0=None,add_count=False,fill_na=False):
    if mean0 is None:
        mean0=df[vn_y].mean()

    #对每条数据进行今天和device_ip结合
    _key_src=np.add(df[key_src].astype('str').values,df[vn].astype('str').values)
    #对每条数据进行昨天和device_ip结合
    _key_tgt=np.add(df[key_tgt].astype('str').values,df[vn].astype('str').values)

    print("aggreate by src key")
    #day
    grp1 = df.groupby(_key_src)
    sum1 = grp1[vn_y].aggregate(np.sum)
    cnt1 = grp1[vn_y].aggregate(np.size)

    print("map to tgt key")
    _sum = sum1[_key_tgt].values
    _cnt = cnt1[_key_tgt].values

    if fill_na:
        print("fill in na")
        _cnt[np.isnan(_cnt)]=0
        _sum[np.isnan(_sum)]=0
    print("calc exp")
    vn_yexp = 'exp_' + vn +'_' + key_src + '_' +key_tgt
    df[vn_yexp] = (_sum + cred_k*mean0)/(_cnt + cred_k)

    if add_count:
        print("add count")
        vn_cnt_src = 'cnt_' + vn + '_' + key_src
        df[vn_cnt_src] = _cnt

        grp2 = df.groupby(_key_tgt)
        cnt2 =grp2[vn_y].aggregate(np.size)
        _cnt2 = cnt2[_key_tgt].values
        vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
        df[vn_cnt_tgt] = _cnt2

def my_grp_cnt(group_by,count_by):
    #lexsort对groupby排序对相同的值用countby排序，返回索引
    #_ord为排序后的deviceip索引值，
    _ord = np.lexsort((count_by,group_by))
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    runnting_cnt = 0
    for i in range(1,group_by.size):
        i0 = _ord[i]
        #i0为排序后的索引值
        #判断当前deviceip是否等于上一个
        if _prev_grp == group_by[i0]:
            #再看上一个id与当前的id是否相同不相同则进行累和
            if count_by[_ord[i-1]] !=count_by[i0]:
                runnting_cnt += 1
        else:
            #若不等于，重新计数
            runnting_cnt = 1
            _prev_grp = group_by[i0]

        #判断是否为最后一个或者下一个不等于当前
        #若判断成立，将count值赋给cs1
        if i==group_by.size-1 or group_by[i0] !=group_by[_ord[i+1]]:
            j=i
            while True:
                j0 = _ord[j]
                #给索引赋值
                _cs1[j0]=runnting_cnt
                if j==0 or group_by[_ord[j-1]] !=group_by[j0]:
                    break
                j-=1
    return _cs1#cs1 count了每个相同deviceip的appid个数
    
def my_grp_idx(group_by,count_by):
    #group_by:device_ip,
    #count_by:str(id)
    #
    _ord = np.lexsort((count_by,group_by))
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'   
    for i in range(1,group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            _cs1[i]=_cs1[i-1] + 1
        else:
            _cs1[i] = 1
            _prev_grp = group_by[i0]
    #_cs1 是排序后的rank值
    #_ord 是排序的索引
    #为了将cs1对应到未排序的表上去，采用以下操作
    #ord_idx是排序的索引的索引
    ord_idx = np.zeros(group_by.size,dtype=np.int)
    ord_idx[_ord] = np.asarray(range(group_by.size))
    #_cs1精髓
    return _cs1[ord_idx]

def get_agg(group_by,value,func):
    #device_ip.values
    #id
    #np.size
    #Series的groupby和dataframe的差别
    #其实这里就是获取groupby的出现次数，value可有可无
    g1 = pd.Series(value).groupby(group_by)
    agg1 = g1.aggregate(func)
    r1 = agg1[group_by].values
    return r1

def cntDualKey(df,vn,vn2,key_src,key_tgt,fill_na=False):
    #vn:device_ip
    #key_src:day_hour
    #key_tgt:day_hour_prev,day_hour,day_hour_next
    _key_src = np.add(df[key_src].astype('str').values,df[vn].astype('str').values)
    _key_tgt = np.add(df[key_tgt].astype('str').values,df[vn].astype('str').values)

    print("aggregate by src key")
    #以_key_src作索引做grouby操作
    #取出vn列(其实取哪列都一样)
    grp1 = df.groupby(_key_src)
    #之所以这里是grp1[vn]如若不加vn则会对整个dataframe进行size操作，效率低下
    cnt1 = grp1[vn].aggregate(np.size)
    _cnt = cnt1[_key_tgt].values

    if fill_na is not None:
        _cnt[np.isnan(_cnt)] = fill_na

    #cnt_device_ip_day_hour_prev
    #cnt_device_ip_day_hour
    #cnt_device_ip_day_hour_next
    vn_cnt_tgt = 'cnt_' + vn +'_' +key_tgt 
    df[vn_cnt_tgt] = _cnt