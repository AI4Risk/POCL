import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,GATConv
import torch.nn as nn
import math

class SAGE(torch.nn.Module):
    def __init__(self,in_feats, n_hidden):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, n_hidden)
        self.conv2 = SAGEConv(n_hidden, n_hidden)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,p=0.5)
        x = self.conv2(x, edge_index)
        return x

class Encoder(nn.Module):  
    def __init__(self, in_feats,n_hidden):
        super(Encoder, self).__init__()
        self.conv = SAGE(
             in_feats,n_hidden
        )
    
    def forward(self,x,edge):
        x = self.conv(x,edge)
        return x

class Decoder(nn.Module):  
    def __init__(self, n_out):
        super(Decoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_out, n_out))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self,x):
        x=torch.matmul(x,self.weight)
        return x
    
class ADDer(nn.Module):
    def __init__(self, in_feats,n_hidden):
        super(ADDer, self).__init__()
        self.encoder=Encoder(in_feats,n_hidden)

        self.decoder=Decoder(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,x,xedge,y,pretrain):
        if(pretrain==True):
            x1=x[0]
            x2=x[1]
            x1edge=xedge[0]
            x2edge=xedge[1]
            x1=self.encoder(x1,x1edge)
            x2=self.encoder(x2,x2edge)
            x1=self.decoder(x1)
            x2=self.decoder(x2)
            l1=self.loss(x1,torch.zeros_like(x1))#0
            l2=self.loss(x2,torch.ones_like(x2))#1
            return l1+l2
        else:
            x=self.encoder(x,xedge)
            dx=self.decoder(x)
            pos=dx[y]
            neg=dx[~y]
            l1=self.loss(neg,torch.zeros_like(neg))#0
            l2=self.loss(pos,torch.ones_like(pos))#1
            return  l1+l2,x
        
class MAS(object):
    def __init__(self, pretrain_model, classifier_model, dataloader, history_importance):
        self.pretrain_model = pretrain_model
        self.classifier_model = classifier_model
        self.dataloader = dataloader

        self.history_importance=history_importance

        # Combine parameters of both models
        self.params = {n: p for n, p in list(self.pretrain_model.named_parameters()) + list(self.classifier_model.named_parameters()) if p.requires_grad} 
        self.p_old = {} 
        self._precision_matrices = self.calculate_importance() 

        for n, p in self.params.items():
          self.p_old[n] = p.clone().detach() 
  
    def calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0) 

        self.classifier_model.eval()
        if self.dataloader is not None:
            num_data = len(self.dataloader)
            for data in self.dataloader:
                self.pretrain_model.zero_grad()
                self.classifier_model.zero_grad()

                output = self.pretrain_model(data[0],data[1])
                output = self.classifier_model(output,data[1])
                output = torch.sqrt(output.pow(2))
                loss = torch.sum(output, dim=1)
                loss = loss.mean()
                loss.backward()

                for n, p in list(self.pretrain_model.named_parameters()) + list(self.classifier_model.named_parameters()):                      
                    precision_matrices[n].data += p.grad.abs() / num_data

            precision_matrices = {n: p for n, p in precision_matrices.items()}

        self.history_importance.append(precision_matrices)

        overall_importance = {}
        m = 0.4
        for n, p in self.params.items():
            overall_importance[n] = p.clone().detach().fill_(0) 
            for i in range(len(self.history_importance)):
                if self.history_importance[i]:
                    overall_importance[n] += m * overall_importance[n] + (1-m) * self.history_importance[i][n]

        min_p = min(p.min().item() for p in overall_importance.values())
        max_p = max(p.max().item() for p in overall_importance.values())

        for n, p in overall_importance.items():
            normalized_p = (p - min_p) / (max_p - min_p + 1e-6)
            overall_importance[n] = normalized_p

        return overall_importance

    def penalty(self, pretrain_model,classifier_model):
        loss = 0
        for n, p in list(pretrain_model.named_parameters()) + list(classifier_model.named_parameters()):
          _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
          loss += _loss.sum()
        return loss
    
    def get_history_importance(self):
        return self.history_importance
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return out
    
class GNN(torch.nn.Module):
    def __init__(self,size,n_out):
        super(GNN,self).__init__()
        self.conv1 = GATConv(size, 8, 8)
        self.conv2 = GATConv(64, 64, 1)
        self.linear= torch.nn.Linear(64,n_out)
    def forward(self,data):
        x,edge_index=data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.linear(x)
        return F.log_softmax(x,dim=1)
    
    
class Classifier(torch.nn.Module):
    def __init__(self,size,n_out):
        super(Classifier,self).__init__()
        self.gat1=GATConv(size,8,8)
        # self.gat2=GATConv(64,64,1)
        self.linear=nn.Linear(64,n_out)
    def forward(self,x,edge_index):
        x=self.gat1(x,edge_index)
        x=F.relu(x)
        x=F.dropout(x,p=0.5)
        # x=self.gat2(x,edge_index)
        x=self.linear(x)
        return x
    
