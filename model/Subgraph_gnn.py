import os
import torch
import torch.nn as nn
from torch.nn import Linear,Dropout,LayerNorm
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,TransformerConv,GINConv,TAGConv,SAGPooling,global_mean_pool
from attention import GlobalAttention_gated
import numpy as np
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self,n_feature,conv_type,fc_scale,droput):
        super(SpatialAttentionModule,self).__init__()
        self.conv_type = conv_type
        self.n_feature = n_feature

        self.fc1 = nn.Linear(2,2*fc_scale)
        self.drop = nn.Dropout(droput)
        self.fc2 = nn.Linear(2*fc_scale,1)

    def FC(self,h):
        h=self.fc1(h)
        h=self.drop(h)
        h=h.tanh()
        h=self.fc2(h)
        h = h.sigmoid()
        return h

    def forward(self,x,edge_index):
        # x : N*Channel*Cluster
        N,C,P = x.shape
        E = edge_index.shape[1]
        avgout = torch.mean(x,dim=2)
        maxout = torch.max(x,dim=2)[0]
        out_avg_max = torch.stack([avgout,maxout],dim=2)
        o = self.FC(o).squeeze(2)
        return o,out_avg_max

class SAG_channel(torch.nn.Module):
    def __init__(self,n_feature=35,hidden_dim=16,SAG_ratio=0.3, n_class=2,drop_out_ratio=0.3,CONV_TYPE='GCN',act_op='relu',before_pooling_layer=1,after_pooling_layer=1,num_K=2):
        super().__init__()

        self.alpha_p = 0.5
        self.alpha_f = 0.5
        self.k = num_K
        self.conv_type = CONV_TYPE
        
        self.before_pooling_conv_layers_list = []
        self.after_pooling_conv_layers_list = []
        assert before_pooling_layer>0

        self.spatialAtt = ChannelAttentionModule(n_feature,self.conv_type)

        if self.conv_type=='GCN':
            self.before_pooling_conv_layers_list.append(GCNConv(n_feature,hidden_dim))
        elif self.conv_type =='SAGE':
            self.before_pooling_conv_layers_list.append(SAGEConv(n_feature,hidden_dim))
        elif self.conv_type =='GAT':
            self.before_pooling_conv_layers_list.append(GATConv(in_channels=n_feature,out_channels=hidden_dim,edge_dim=1))
        elif self.conv_type =='TransformerConv':
            self.before_pooling_conv_layers_list.append(TransformerConv(in_channels=n_feature,out_channels=hidden_dim,edge_dim=1))
        elif self.conv_type =='TAGConv':
            self.before_pooling_conv_layers_list.append(TAGConv(in_channels=n_feature,out_channels=hidden_dim))

        for i in range(1,before_pooling_layer):
            if self.conv_type=='GCN':
                self.before_pooling_conv_layers_list.append(GCNConv(hidden_dim,hidden_dim))
            elif self.conv_type =='SAGE':
                self.before_pooling_conv_layers_list.append(SAGEConv(hidden_dim,hidden_dim))
            elif self.conv_type =='GAT':
                self.before_pooling_conv_layers_list.append(GATConv(in_channels=hidden_dim,out_channels=hidden_dim,edge_dim=1))
            elif self.conv_type =='TransformerConv':
                self.before_pooling_conv_layers_list.append(TransformerConv(in_channels=hidden_dim,out_channels=hidden_dim,edge_dim=1))
            elif self.conv_type =='TAGConv':
                self.before_pooling_conv_layers_list.append(TAGConv(in_channels=hidden_dim,out_channels=hidden_dim))

        if self.conv_type=='GCN':
            self.after_pooling_conv_layers_list.append(GCNConv(hidden_dim*before_pooling_layer,hidden_dim))
        elif self.conv_type =='SAGE':
            self.after_pooling_conv_layers_list.append(SAGEConv(hidden_dim*before_pooling_layer,hidden_dim))
        elif self.conv_type =='GAT':
            self.after_pooling_conv_layers_list.append(GATConv(in_channels=hidden_dim*before_pooling_layer,out_channels=hidden_dim,edge_dim=1))
        elif self.conv_type =='TransformerConv':
            self.after_pooling_conv_layers_list.append(TransformerConv(in_channels=hidden_dim*before_pooling_layer,out_channels=hidden_dim,edge_dim=1))
        elif self.conv_type =='TAGConv':
            self.after_pooling_conv_layers_list.append(TAGConv(in_channels=hidden_dim*before_pooling_layer,out_channels=hidden_dim))

        for i in range(1,after_pooling_layer):
            if self.conv_type=='GCN':
                self.after_pooling_conv_layers_list.append(GCNConv(hidden_dim,hidden_dim))
            elif self.conv_type =='SAGE':
                self.after_pooling_conv_layers_list.append(SAGEConv(hidden_dim,hidden_dim))
            elif self.conv_type =='GAT':
                self.after_pooling_conv_layers_list.append(GATConv(in_channels=hidden_dim,out_channels=hidden_dim,edge_dim=1))
            elif self.conv_type =='TransformerConv':
                self.after_pooling_conv_layers_list.append(TransformerConv(in_channels=hidden_dim,out_channels=hidden_dim,edge_dim=1))
            elif self.conv_type =='TAGConv':
                self.after_pooling_conv_layers_list.append(TAGConv(in_channels=hidden_dim,out_channels=hidden_dim))

        self.before_pooling_conv_layers = nn.ModuleList(self.before_pooling_conv_layers_list)
        self.after_pooling_conv_layers = nn.ModuleList(self.after_pooling_conv_layers_list)
        
        self.pool_num = SAG_ratio
        self.act_op = act_op
        self.pool=SAGPooling(hidden_dim*before_pooling_layer,ratio=SAG_ratio)
        self.norm=LayerNorm(hidden_dim*after_pooling_layer)
        self.fc1=Linear(hidden_dim*after_pooling_layer,(hidden_dim)//2)
        self.drop1=Dropout(drop_out_ratio)
        self.fc2=Linear((hidden_dim)//2,n_class)
        self.GAP = GlobalAttention_gated(gate_nn = nn.Sequential(nn.Linear(hidden_dim*after_pooling_layer,hidden_dim*after_pooling_layer//2),nn.BatchNorm1d(hidden_dim*after_pooling_layer//2),nn.ReLU(),nn.Linear(hidden_dim*after_pooling_layer//2,1)))

    def act_func(self,h,act_op):
        if act_op=='relu':
            h = h.relu()
        elif act_op=='tanh':
            h = h.tanh()
        elif act_op=='softmax':
            h = h.softmax(dim=-1)
        elif act_op=='sigmoid':
            h = h.sigmoid()
        elif act_op=='leakyrelu':
            op = nn.LeakyReLU(1e-2)
            h = op(h)
        return h

    def get_dis_matrix(self,pos,pos_):
        x_dis = pos[:,0:1].repeat(1,len(pos_)) - pos_[:,0:1].view(1,-1).repeat(len(pos),1)
        y_dis = pos[:,1:2].repeat(1,len(pos_)) - pos_[:,1:2].view(1,-1).repeat(len(pos),1)
        dis = torch.sqrt(x_dis **2 +y_dis **2)

        dis_norm = dis/torch.max(dis).item()
        dis_norm_score = 2/(dis_norm+1)-1
        return dis_norm_score  # [N*C]
    
    def get_fea_matrix(self,fea,fea_):
        f_dis = None 
        for i in range(fea.shape[1]):
            if f_dis is None:
                f_dis = (fea[:,i:i+1].repeat(1,len(fea_)) - fea_[:,i:i+1].view(1,-1).repeat(len(fea),1)) **2
            else:
                f_dis = f_dis+ (fea[:,i:i+1].repeat(1,len(fea_)) - fea_[:,i:i+1].view(1,-1).repeat(len(fea),1)) **2
        
        f_dis = torch.sqrt(f_dis)
        f_dis_norm = f_dis/torch.max(f_dis).item()
        f_dis_norm_score = 2/(f_dis_norm+1)-1
        return f_dis_norm_score # [N*C]
    
    def MLP(self,h):
        h=self.fc1(h)
        h=self.drop1(h)
        h=h.tanh()
        h=self.fc2(h)
        h = F.softmax(h,dim=-1)

    def H_GNN(self,x,edge_index,edge_weight,conv_layers):
        h_ = None
        for i,c in enumerate(conv_layers):
            if self.conv_type in ['GCN','SAGE']:
                x = c(x,edge_index)
            elif self.conv_type in ['GAT','TransformerConv']:
                x = c(x,edge_index,edge_attr=edge_weight)
            elif self.conv_type in ['TAGConv']:
                x = c(x,edge_index,edge_weight=edge_weight)
            x = self.act_func(x,self.act_op)
            if h_ is None:
                h_ = x
            else:
                h_ = torch.cat([h_1,x],axis=1)
        
        return h_ 

    def forward(self,x_origin,edge_index,edge_weight,pos):
        edge_index_origin = edge_index
        h = x_origin
        
        h_1 = self.H_GNN(x_origin,edge_index_origin,edge_weight,self.before_pooling_conv_layers)
        h, edge_index, edge_weight, batch, perm, score=self.pool(h_1,edge_index,edge_attr = edge_weight)
        pos_ = pos[perm]

        dis_p = self.get_dis_matrix(pos,pos_).detach()
        dis_f = self.get_fea_matrix(h_1, h_1[perm]).detach()
        S = self.alpha_p*dis_p+self.alpha_f*dis_f # N * (N*pool_ratio)
        h = torch.mm(S.transpose(0,1),h_1) 
        origin_dis = torch.argmax(S,dim=1)
        

        channel_A,avg_max = self.channelAtt((x_origin.unsqueeze(2))*(S.unsqueeze(1)),edge_index_origin) # channel_Attention

        h = h + torch.mm(S.transpose(0,1),channel_A*x_origin) 
           
        h_2 = self.H_GNN(h,edge_index,edge_weight,self.after_pooling_conv_layers)    
        h_emb = h_2
        n_h = h_2.size(0)
        batch = edge_index.new_zeros((n_h,1)).flatten()
        gate,h_3 = self.GAP(h_2,batch,gate_act='sigmoid')
        h_3 = h_3/gate.shape[0]
        gate = gate.squeeze(1) 

        vals,indices = torch.topk(gate,len(gate))
        vals_k,indices_k = torch.topk(gate,self.k)
        h_f = self.norm(h_3)
        fea_top = h_f
        h_f = self.MLP(h_f)

        in_indices = torch.zeros(len(indices),dtype=torch.int).cuda()
        for i in range(indices.max()+1):
            in_indices[indices[i]]=i
        mask_points = in_indices[origin_dis]
        
        return h_f,mask_points,h_emb[indices],vals,fea_top,pos_[indices],h_emb[indices_k],channel_A.detach(),avg_max.detach()


class SAG(torch.nn.Module):
    def __init__(self,n_feature=35,hidden_dim=16,SAG_ratio=0.3, n_class=2,drop_out_ratio=0.3,CONV_TYPE='GCN',act_op='relu',before_pooling_layer=1,after_pooling_layer=1,num_K=2):
        super().__init__()

        self.alpha_p = 0.5
        self.alpha_f = 0.5
        self.k = num_K
        self.conv_type = CONV_TYPE
        
        self.before_pooling_conv_layers_list = []
        self.after_pooling_conv_layers_list = []
        assert before_pooling_layer>0

        if self.conv_type=='GCN':
            self.before_pooling_conv_layers_list.append(GCNConv(n_feature,hidden_dim))
        elif self.conv_type =='SAGE':
            self.before_pooling_conv_layers_list.append(SAGEConv(n_feature,hidden_dim))
        elif self.conv_type =='GAT':
            self.before_pooling_conv_layers_list.append(GATConv(in_channels=n_feature,out_channels=hidden_dim,edge_dim=1))
        elif self.conv_type =='TransformerConv':
            self.before_pooling_conv_layers_list.append(TransformerConv(in_channels=n_feature,out_channels=hidden_dim,edge_dim=1))
        elif self.conv_type =='TAGConv':
            self.before_pooling_conv_layers_list.append(TAGConv(in_channels=n_feature,out_channels=hidden_dim))

        for i in range(1,before_pooling_layer):
            if self.conv_type=='GCN':
                self.before_pooling_conv_layers_list.append(GCNConv(hidden_dim,hidden_dim))
            elif self.conv_type =='SAGE':
                self.before_pooling_conv_layers_list.append(SAGEConv(hidden_dim,hidden_dim))
            elif self.conv_type =='GAT':
                self.before_pooling_conv_layers_list.append(GATConv(in_channels=hidden_dim,out_channels=hidden_dim,edge_dim=1))
            elif self.conv_type =='TransformerConv':
                self.before_pooling_conv_layers_list.append(TransformerConv(in_channels=hidden_dim,out_channels=hidden_dim,edge_dim=1))
            elif self.conv_type =='TAGConv':
                self.before_pooling_conv_layers_list.append(TAGConv(in_channels=hidden_dim,out_channels=hidden_dim))

        if self.conv_type=='GCN':
            self.after_pooling_conv_layers_list.append(GCNConv(hidden_dim*before_pooling_layer,hidden_dim))
        elif self.conv_type =='SAGE':
            self.after_pooling_conv_layers_list.append(SAGEConv(hidden_dim*before_pooling_layer,hidden_dim))
        elif self.conv_type =='GAT':
            self.after_pooling_conv_layers_list.append(GATConv(in_channels=hidden_dim*before_pooling_layer,out_channels=hidden_dim,edge_dim=1))
        elif self.conv_type =='TransformerConv':
            self.after_pooling_conv_layers_list.append(TransformerConv(in_channels=hidden_dim*before_pooling_layer,out_channels=hidden_dim,edge_dim=1))
        elif self.conv_type =='TAGConv':
            self.after_pooling_conv_layers_list.append(TAGConv(in_channels=hidden_dim*before_pooling_layer,out_channels=hidden_dim))

        for i in range(1,after_pooling_layer):
            if self.conv_type=='GCN':
                self.after_pooling_conv_layers_list.append(GCNConv(hidden_dim,hidden_dim))
            elif self.conv_type =='SAGE':
                self.after_pooling_conv_layers_list.append(SAGEConv(hidden_dim,hidden_dim))
            elif self.conv_type =='GAT':
                self.after_pooling_conv_layers_list.append(GATConv(in_channels=hidden_dim,out_channels=hidden_dim,edge_dim=1))
            elif self.conv_type =='TransformerConv':
                self.after_pooling_conv_layers_list.append(TransformerConv(in_channels=hidden_dim,out_channels=hidden_dim,edge_dim=1))
            elif self.conv_type =='TAGConv':
                self.after_pooling_conv_layers_list.append(TAGConv(in_channels=hidden_dim,out_channels=hidden_dim))

        self.before_pooling_conv_layers = nn.ModuleList(self.before_pooling_conv_layers_list)
        self.after_pooling_conv_layers = nn.ModuleList(self.after_pooling_conv_layers_list)
        
        self.pool_num = SAG_ratio
        self.act_op = act_op
        self.pool=SAGPooling(hidden_dim*before_pooling_layer,ratio=SAG_ratio)
        self.norm=LayerNorm(hidden_dim*after_pooling_layer)
        self.fc1=Linear(hidden_dim*after_pooling_layer,(hidden_dim)//2)
        self.drop1=Dropout(drop_out_ratio)
        self.fc2=Linear((hidden_dim)//2,n_class)
        self.GAP = GlobalAttention_gated(gate_nn = nn.Sequential(nn.Linear(hidden_dim*after_pooling_layer,hidden_dim*after_pooling_layer//2),nn.BatchNorm1d(hidden_dim*after_pooling_layer//2),nn.ReLU(),nn.Linear(hidden_dim*after_pooling_layer//2,1)))

    def act_func(self,h,act_op):
        if act_op=='relu':
            h = h.relu()
        elif act_op=='tanh':
            h = h.tanh()
        elif act_op=='softmax':
            h = h.softmax(dim=-1)
        elif act_op=='sigmoid':
            h = h.sigmoid()
        elif act_op=='leakyrelu':
            op = nn.LeakyReLU(1e-2)
            h = op(h)
        return h

    def get_dis_matrix(self,pos,pos_):
        x_dis = pos[:,0:1].repeat(1,len(pos_)) - pos_[:,0:1].view(1,-1).repeat(len(pos),1)
        y_dis = pos[:,1:2].repeat(1,len(pos_)) - pos_[:,1:2].view(1,-1).repeat(len(pos),1)
        dis = torch.sqrt(x_dis **2 +y_dis **2)

        dis_norm = dis/torch.max(dis).item()
        dis_norm_score = 2/(dis_norm+1)-1
        return dis_norm_score  # [N*C]
    
    def get_fea_matrix(self,fea,fea_):
        f_dis = None 
        for i in range(fea.shape[1]):
            if f_dis is None:
                f_dis = (fea[:,i:i+1].repeat(1,len(fea_)) - fea_[:,i:i+1].view(1,-1).repeat(len(fea),1)) **2
            else:
                f_dis = f_dis+ (fea[:,i:i+1].repeat(1,len(fea_)) - fea_[:,i:i+1].view(1,-1).repeat(len(fea),1)) **2
        
        f_dis = torch.sqrt(f_dis)
        f_dis_norm = f_dis/torch.max(f_dis).item()
        f_dis_norm_score = 2/(f_dis_norm+1)-1
        return f_dis_norm_score # [N*C]
    
    def MLP(self,h):
        h=self.fc1(h)
        h=self.drop1(h)
        h=h.tanh()
        h=self.fc2(h)
        h = F.softmax(h,dim=-1)

    def H_GNN(self,x,edge_index,edge_weight,conv_layers): # Hierarchical graph convolution
        h_ = None
        for i,c in enumerate(conv_layers):
            if self.conv_type in ['GCN','SAGE']:
                x = c(x,edge_index)
            elif self.conv_type in ['GAT','TransformerConv']:
                x = c(x,edge_index,edge_attr=edge_weight)
            elif self.conv_type in ['TAGConv']:
                x = c(x,edge_index,edge_weight=edge_weight)
            x = self.act_func(x,self.act_op)
            if h_ is None:
                h_ = x
            else:
                h_ = torch.cat([h_1,x],axis=1)
        
        return h_ 

    def forward(self,x_origin,edge_index,edge_weight,pos):
        edge_index_origin = edge_index
        h = x_origin
        
        h_1 = self.H_GNN(x_origin,edge_index_origin,edge_weight,self.before_pooling_conv_layers)
        h, edge_index, edge_weight, batch, perm, score=self.pool(h_1,edge_index,edge_attr = edge_weight)
        pos_ = pos[perm]

        dis_p = self.get_dis_matrix(pos,pos_).detach()
        dis_f = self.get_fea_matrix(h_1, h_1[perm]).detach()
        S = self.alpha_p*dis_p+self.alpha_f*dis_f # N * (N*pool_ratio)
        h = torch.mm(S.transpose(0,1),h_1) 
        origin_dis = torch.argmax(S,dim=1)
           
        h_2 = self.H_GNN(h,edge_index,edge_weight,self.after_pooling_conv_layers)    
        h_emb = h_2
        n_h = h_2.size(0)
        batch = edge_index.new_zeros((n_h,1)).flatten()
        gate,h_3 = self.GAP(h_2,batch,gate_act='sigmoid')
        h_3 = h_3/gate.shape[0]
        gate = gate.squeeze(1) 

        vals,indices = torch.topk(gate,len(gate))
        vals_k,indices_k = torch.topk(gate,self.k)
        h_f = self.norm(h_3)
        fea_top = h_f
        h_f = self.MLP(h_f)

        in_indices = torch.zeros(len(indices),dtype=torch.int).cuda()
        for i in range(indices.max()+1):
            in_indices[indices[i]]=i
        mask_points = in_indices[origin_dis]
        
        return h_f,mask_points,h_emb[indices],vals,fea_top,pos_[indices],h_emb[indices_k]