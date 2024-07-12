import pandas as pd
import os
from sklearn import preprocessing
import numpy as np
data_route="./Dataset/"

bene=pd.read_csv(data_route+'bene.csv')#获取数据
claims=pd.read_csv(data_route+'claims.csv')
data_join=pd.merge(claims,bene,on='BeneID',how='left')#合并数据
#data_join['ClaimStartDt']=pd.to_datetime(data_join['ClaimStartDt'])
print(data_join['ClaimStartDt'].max())
print(data_join['ClaimStartDt'].min())

#定义一个函数将column中的日期，转化为特征，分别存入列column_year,column_month,column_day中
def date_to_feature(data,column):
    column_year=column+'_year'
    column_month=column+'_month'
    column_day=column+'_day'
    data[column_year]=data[column].dt.year
    data[column_month]=data[column].dt.month
    data[column_day]=data[column].dt.day
    return data

#接下来将ClaimStartDt和ClaimEndDt相差的天数存入新列DateDiff中，并删除ClaimEndDt
data_join['DateDiff']=pd.to_datetime(data_join['ClaimEndDt'])-pd.to_datetime(data_join['ClaimStartDt'])
data_join['DateDiff']=data_join['DateDiff'].dt.days
data_join.drop(['ClaimEndDt'],axis=1,inplace=True)

# #接下来将AdmissionDt和DischargeDt相差的天数存入新列AdmissionDiff中
data_join['AdmissionDiff']=pd.to_datetime(data_join['DischargeDt'])-pd.to_datetime(data_join['AdmissionDt'])
data_join['AdmissionDiff']=data_join['AdmissionDiff'].dt.days

# #接下来将DOD检查是否为-1，如果不是则改为1，是则改为0
data_join['DOD']=data_join['DOD'].apply(lambda x: 0 if x=='-1' else 1)
# #将这些列的值全部-1
need_minusone=['ChronicCond_Alzheimer','ChronicCond_Heartfailure',
'ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary','ChronicCond_Depression',
'ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis','ChronicCond_stroke','Gender','Race']
for i in need_minusone:
    data_join[i]=data_join[i].apply(lambda x:x-1)
need_10000minusone=['ClmAdmitDiagnosisCode','DiagnosisGroupCode']
for i in need_10000minusone:
    data_join[i]=data_join[i].apply(lambda x:x-10000)

# #将AdmissionDt和DischargeDt删除
data_join.drop(['AdmissionDt','DischargeDt'],axis=1,inplace=True)

# #利用date_to_feature函数，将DOB转化成特征，删除DOB
data_join['DOB']=pd.to_datetime(data_join['DOB'])
data_join=date_to_feature(data_join,'DOB')
data_join.drop(['DOB'],axis=1,inplace=True)
data_join.drop(['DOB_day','PotentialGroupFraud'],axis=1,inplace=True)
data_join['ClaimStartDt']=pd.to_datetime(data_join['ClaimStartDt'])

#数据按照1天分组，按照ClaimStartDt为起始周期
de_day=1
next_data=[]
start_date=pd.to_datetime('2009-01-01')
while(start_date<pd.to_datetime('2009-12-31')):
    temp=data_join[(data_join['ClaimStartDt']>=start_date)&(data_join['ClaimStartDt']<start_date+pd.Timedelta(days=de_day))]
    #print(start_date,len(temp))
    next_data.append(temp)
    start_date=start_date+pd.Timedelta(days=de_day)
    

#根据输入的数据，根据条件建立边
def create_edges(data,large=0):
    data=data.reset_index(drop=True)
    #建立边的列表
    edges=[]
    #建立边的字典，边的定义方式为存在相同的BeneID或者Provider
    datagroup=data.groupby(['BeneID'])
    for i in datagroup:
        if(len(i[1])>1):
            #如果是large，则还要判断是不是5天内的节点，是则对两两取索引
            index=i[1].index
            for j in range(len(index)):
                indexa=index[j]
                for k in range(j+1,len(index)):
                    indexb=index[k]
                    if((large==0) or i[1].iloc[j]['ClaimStartDt']-i[1].iloc[k]['ClaimStartDt']<pd.Timedelta(days=5)):
                        if(indexa<indexb):
                            edges.append((indexa,indexb))
                        else:
                            edges.append((indexb,indexa))


    providergroup=data.groupby(['Provider'])
    for i in providergroup:
        if(len(i[1])>1):
            #如果是large，则还要判断是不是5天内的节点，是则对两两取索引
            index=i[1].index
            for j in range(len(index)):
                indexa=index[j]
                for k in range(j+1,len(index)):
                    indexb=index[k]
                    if((large==0)or i[1].iloc[j]['ClaimStartDt']-i[1].iloc[k]['ClaimStartDt']<pd.Timedelta(days=5)):
                        if(indexa<indexb):
                            edges.append((indexa,indexb))
                        else:
                            edges.append((indexb,indexa))
    return list(set(edges))

#将边转化形式
def edge_transform(edges):
    edges=np.array(edges)
    edges=edges.T.reshape(2,-1)
    return edges
#将特征pandas转为numpy,并归一化
def pandas_to_numpy(data):
    data=data.reset_index(drop=True)
    data=data.drop(['BeneID','Provider','ClaimStartDt','ClaimID'],axis=1)
    data=data.values
    data=preprocessing.MinMaxScaler().fit_transform(data)
    return data

def turn_to_no_direction(edge):
    #将边转化为无向图
    edge_begin=edge[0]
    edge_end=edge[1]
    begin_new_begin=np.concatenate((edge_begin,edge_end))
    begin_new_end=np.concatenate((edge_end,edge_begin))
    return np.array([begin_new_begin,begin_new_end])


import copy
tempnext_data = copy.deepcopy(next_data)
dataset=[]
for i in range(len(next_data)):
    next_data_edge=create_edges(next_data[i],0)
    next_data_edge=edge_transform(next_data_edge)
    next_data[i]=pandas_to_numpy(next_data[i])
    dataset.append((next_data[i],next_data_edge))
print('next_data_edge finish')
print(len(dataset))

for i in range(len(dataset)):#全部转为无向图
    dataset[i]=(dataset[i][0],turn_to_no_direction(dataset[i][1]))
print(len(dataset[0][1][0]))

import pickle
with open(data_route+"datasetonline.dat", "wb") as file:
    pickle.dump(dataset, file)

dataset=[]
next_data = tempnext_data
for i in next_data:
    data_all=[i[i['PotentialFraud']==False],i[i['PotentialFraud']==True]]
    for j in data_all:
        next_data_edge=create_edges(j,0)
        next_data_edge=edge_transform(next_data_edge)
        j=pandas_to_numpy(j)
        dataset.append((j,next_data_edge))
print('next_data_edge finish')
print(len(dataset))

for i in range(len(dataset)):#全部转为无向图
    dataset[i]=(dataset[i][0],turn_to_no_direction(dataset[i][1]))
print(len(dataset[0][1][0]))

with open(data_route+"datasettwo.dat", "wb") as file:
    pickle.dump(dataset, file)