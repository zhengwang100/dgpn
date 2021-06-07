import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_functions import dot_sim, use_cuda

''' 
Decomposed Graph Prototype Network (DGPN)
 --- At the firt layer, we decompose a k-hop gcn layer to {k+1} parts
 --- At the second-last layer, we use a fc layer to map the local and global embeddings to pred the csd_matrix. 
'''
device = use_cuda()

class DGPN(nn.Module):
    def __init__(self, n_in, n_h, dropout):
        super(DGPN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h, bias=True)
        self.fc_local_pred_csd = nn.Linear(n_h, n_h, bias=True) 
        self.fc_final_pred_csd = nn.Linear(n_h, n_h, bias=True)  # used for the last layer
        
        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
    
    def forward(self, feature_list, csdmatrix):
        # the local item: 1. get k-hop-gcn by one layer; 2. get local loss
        templist = []
        local_pred_result_list = []
        for features in feature_list:
            temp_embedds = self.fc1(features.to(device))
            temp_embedds = self.act(temp_embedds)
            temp_embedds = F.dropout(temp_embedds, p=self.dropout, training=self.training)
            templist.append(temp_embedds)
            local_pred_csd = self.fc_local_pred_csd(temp_embedds)
            local_pred_result_list.append(dot_sim(local_pred_csd, csdmatrix.t()))
        
        total_embedds = torch.sum(torch.stack(templist), dim=0)
        total_embedds_pred_csd = self.fc_final_pred_csd(total_embedds)
        total_embedds_pred_csd = F.dropout(total_embedds_pred_csd, p=self.dropout, training=self.training)
        preds = dot_sim(total_embedds_pred_csd, csdmatrix.t())

        return preds, local_pred_result_list