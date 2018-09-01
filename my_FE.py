import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
import time
import xgboost as xgb
from sklearn.externals.joblib import dump, load, Parallel, delayed
import utils_my
from utils_my import *

t0org0=pd.read_csv("data/train.csv")
h0org=pd.read_csv("data/test.csv")

h0org['click']=0
t0org=pd.concat([t0org0,h0org])
print("finshed loading raw data and concat")

print("add some basic features")
t0org['day']=np.round(t0org0.hour%10000 /100)
t0org['hour1'] = np.round(t0org.hour % 100)#hour时
t0org['day_hour'] = (t0org.day.values - 21) * 24 + t0org.hour1.values#距离21号0时的时间数
t0org['day_hour_prev'] = t0org['day_hour'] - 1#前一小时
t0org['day_hour_next'] = t0org['day_hour'] + 1#后一小时
t0org['app_or_web'] = 0
#appid为这个值大量出现表示apporweb
t0org.ix[t0org.app_id.values=='ecad2386', 'app_or_web'] = 1

t0 = t0org
t0['app_site_id'] = np.add(t0.app_id.values, t0.site_id.values)

print("to count prev/current/next hour by ip ...")
#count某用户上一小时的浏览数
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour_prev', fill_na=0)
#count某用户这一小时的浏览数
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour', fill_na=0)
#count某用户下一小时的浏览数
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour_next', fill_na=0)

#创建day之间的差异
print("to create day diffs")
#创建每条数据上一天的索引
t0['pday']=t0.day-1
calcDualKey(t0, 'device_ip', None, 'day', 'pday', 'click', 10, None, True, True)
t0['cnt_diff_device_ip_day_pday'] = t0.cnt_device_ip_day.values - t0.cnt_device_ip_pday.values
#hour & web 
t0['hour1_web'] = t0.hour1.values
t0.ix[t0.app_or_web.values==0,'hour1_web'] = -1
#通过deviceip来计数appid个数
t0['app_cnt_by_dev_ip'] = my_grp_cnt(t0.device_ip.values.astype('str'), t0.app_id.values.astype('str'))
#rank根据id来排序表达出现次数
t0['rank_dev_ip'] = my_grp_idx(t0.device_ip.values.astype('str'), t0.id.values.astype('str'))
t0['rank_day_dev_ip'] = my_grp_idx(np.add(t0.device_ip.values, t0.day.astype('str').values).astype('str'), t0.id.values.astype('str'))
t0['rank_app_dev_ip'] = my_grp_idx(np.add(t0.device_ip.values, t0.app_id.values).astype('str'), t0.id.values.astype('str'))
#获取devip和devid的计数值
t0['cnt_dev_ip'] = get_agg(t0.device_ip.values, t0.id, np.size)
t0['cnt_dev_id'] = get_agg(t0.device_id.values, t0.id, np.size)
#不理解这两个特征的存在
#将计数特征的数字进行控制使得上限为300
t0['dev_id_cnt2'] = np.minimum(t0.cnt_dev_id.astype('int32').values, 300)
t0['dev_ip_cnt2'] = np.minimum(t0.cnt_dev_ip.astype('int32').values, 300)
#创建了id和ip两个特征并把频率为1的数据全部置为同一值_only1
t0['dev_id2plus'] = t0.device_id.values
t0.ix[t0.cnt_dev_id.values == 1, 'dev_id2plus'] = '___only1'
t0['dev_ip2plus'] = t0.device_ip.values
t0.ix[t0.cnt_dev_ip.values == 1, 'dev_ip2plus'] = '___only1'
#创建特征：这一小时和上一小时以及下一小时的浏览数差，暂时没理解为什么乘以apporweb
t0['diff_cnt_dev_ip_hour_phour_aw2_prev'] = (t0.cnt_device_ip_day_hour.values - t0.cnt_device_ip_day_hour_prev.values) * ((t0.app_or_web * 2 - 1)) 
t0['diff_cnt_dev_ip_hour_phour_aw2_next'] = (t0.cnt_device_ip_day_hour.values - t0.cnt_device_ip_day_hour_next.values) * ((t0.app_or_web * 2 - 1)) 


#利用joblib保存t0对象
print("to save t0 ....")
dump(t0,"data/t0.joblib_dat")

print("to generate t0tv_mx....")

app_or_web = None
_start_day = 22
list_param = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type']
feature_list_dict = {}

feature_list_name = 'tvexp3'
feature_list_dict[feature_list_name] = list_param + \
                            ['cnt_diff_device_ip_day_pday','hour1_web','cnt_dev_ip','cnt_dev_id',
                            'app_cnt_by_dev_ip', 'app_or_web','cnt_device_ip_day_hour',
                            'diff_cnt_dev_ip_hour_phour_aw2_prev',
                            'diff_cnt_dev_ip_hour_phour_aw2_next',
                            'rank_dev_ip','rank_day_dev_ip','rank_app_dev_ip',]
    
#filter 22->30
#filter_tv = np.logical_and(t0.day.values >= _start_day,t0.day.values < 31)
#22-》29
#filter_t1 = np.logical_and(t0.day.values < 30,filter_tv)
#30
#filter_v1 = np.logical_and(~filter_t1,filter_tv)

for vn in feature_list_dict[feature_list_name]:
     if vn not in t0.columns:
         print("="*60+vn)

t0tv_mx = t0[feature_list_dict[feature_list_name]].values

print("to save t0tv_mx.....")

t0tv_mx_save={}
t0tv_mx_save['t0tv_mx'] = t0tv_mx
t0tv_mx_save['click'] = t0.click.values
t0tv_mx_save['day'] = t0.day.values
t0tv_mx_save['site_id'] = t0.site_id.values
dump(t0tv_mx_save,"data/t0tv_mx.joblib_dat")