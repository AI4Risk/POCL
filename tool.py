
import torch
import pickle
import numpy as np

data_route="./Dataset/"
def get_online_dataset():
    with open(data_route+'datasetonline.dat','rb') as f:
        dataset=pickle.load(f)
    clear_dataset=[]
    for i in dataset:
        element=i[0]
        edges=i[1]
        y = element[:, 5]
        element = np.delete(element, 5, axis=1)
        element = torch.tensor(element, dtype=torch.float)
        y = torch.tensor(y.T, dtype=torch.long)
        edges = torch.LongTensor(edges).long()
        y= np.array(y,dtype=bool)
        clear_dataset.append([element,edges,y])
    return clear_dataset

def get_contrast_dataset():
    with open(data_route+'datasettwo.dat','rb') as f:
        datasettwo=pickle.load(f)
    clear_datasettwo=[]
    for i in datasettwo:
        element=i[0]
        edges=i[1]
        y = element[:, 5]
        element = np.delete(element, 5, axis=1)
        element = torch.tensor(element, dtype=torch.float)
        y = torch.tensor(y.T, dtype=torch.bool)
        edges = torch.LongTensor(edges).long()
        clear_datasettwo.append([element,edges,y])
    return clear_datasettwo

def get_avg(source):
    averages = []
    total = 0
    count = 0
    
    for value in source:
        total += value
        count += 1

        if count % 30 == 0:
            averages.append(total / count)
            total = 0
            count = 0

    if count > 0:
        averages.append(total / count)
    
    return averages