from ast import Num
import os
import argparse
import setproctitle

parser = argparse.ArgumentParser()
parser.add_argument('--sag_r', default=64, type=int,
                    help='config file path')
parser.add_argument('--gpu_id', default='7', type=str,
                    help='')
parser.add_argument('--fold_s', default=0, type=int,
                    help='')
parser.add_argument('--fold_e', default=35, type=int,
                    help='')
parser.add_argument('--repeat_s', default=0, type=int,
                    help='')
parser.add_argument('--repeat_e', default=1, type=int,
                    help='')                    
parser.add_argument('--convtype', default='SAGE', type=str,
                    help='')
parser.add_argument('--act_op', default='relu', type=str,
                    help='')
parser.add_argument('--hd', default=64, type=int,
                    help='')
parser.add_argument('--before_layer', default=1, type=int,
                    help='')
parser.add_argument('--after_layer', default=1, type=int,
                    help='')
parser.add_argument('--epoch', default=45, type=int,
                    help='')
parser.add_argument('--early_stop', default=20, type=int,
                    help='')
parser.add_argument('--lr', default=7e-4, type=float,
                    help='')
parser.add_argument('--weight_decay', default=2e-5, type=float,
                    help='')
parser.add_argument('--batch_size', default=16, type=int,
                    help='')
parser.add_argument('--lsim_loss', default=0.5, type=float,
                    help='')
parser.add_argument('--dropout', default=0.3, type=float,
                    help='')
parser.add_argument('--pool_type', default='sagpool', type=str,
                    help='')  
parser.add_argument('--Ks', default=2, type=int,
                    help='')
parser.add_argument('--Ke', default=10, type=int,
                    help='')
parser.add_argument('--K_step', default=2, type=int,
                    help='')  
parser.add_argument('--ckpt_save_epoch', default=25, type=int,
                    help='')                                       
parser.add_argument('--ckpt_path', default='checkpoint', type=str,
                    help='')  
parser.add_argument('--res_path', default='log_res', type=str,
                    help='')  
parser.add_argument('--gnn_path', default='../data/melanoma/gnn_data', type=str,
                    help='')  
parser.add_argument('--label_path', default='../data/melanoma/label_and_fold/response_label_dict.pkl', type=str,
                    help='')  
parser.add_argument('--fold_path', default='../data/melanoma/label_and_fold/leave_one_fold_for_response.pkl', type=str,
                    help='')  


config = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.nn import Linear,Dropout,LayerNorm
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,TransformerConv,GINConv,TAGConv,SAGPooling,global_mean_pool,GlobalAttention_gated
import numpy as np
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from torch_geometric.data import Data
from model import SAG
from loss import *
from metrics import * 
from utils import *


device=torch.device("cuda:0")
epoch = config.epoch
early_stop = config.early_stop
batch_size = config.batch_size
lsim_loss_lambda = config.lsim_loss


for ki in range(config.Ks,config.Ke,config.K_step):
    for drop_out_r in [config.dropout]:
        for conv_type in [config.convtype]:
            for sag_r in [1/config.sag_r]:
                for hd in [config.hd]:
                    trial_test_fold_mean = []
                    os.makedirs(config.res_path,exist_ok=True)
                    os.makedirs(config.ckpt_path,exist_ok=True)
                    os.makedirs(os.path.join(config.res_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer)),exist_ok=True)
                    os.makedirs(os.path.join(config.res_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}/subgraph'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer)),exist_ok=True)
                    os.makedirs(os.path.join(config.ckpt_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer)),exist_ok=True)
                    
                    os.makedirs(config.init_model_path,exist_ok=True)
                    for t in range(config.repeat_s,config.repeat_e):

                        for fi in range(config.fold_s,config.fold_e):
                            f_name=os.path.join(config.res_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer)+'/saved_SAG_GNN_repeat_'+str(t)+'fold'+str(fi)+'.log')
                            model_pathname = os.path.join(config.ckpt_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}_leave_one.pth'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer))

                            f_ = open(f_name,'w')
                            f_.truncate()
                            log_print(f_name,'hidden_dim:{} SAG Pooling_ratio:{} conv_type:{} drop_out:{}'.format(hd,sag_r,conv_type,drop_out_r))
                            log_print(f_name,"=======================FOLD {}=======================".format(fi+1))
                            dataloader = get_dataloader(index=fi,x_path = config.gnn_path,y_path = config.label_path, fold_path = config.fold_path)

                            model=SAG_channel(hidden_dim=hd,SAG_ratio=sag_r,CONV_TYPE=conv_type,act_op=config.act_op,before_pooling_layer=config.before_layer,after_pooling_layer=config.after_layer,num_K=ki).to(device)
                            if t==0 and fi==0:
                                torch.save(model.state_dict(),os.path.join(config.init_model_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}.pth'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer))
                            else:
                                model.load_state_dict(torch.load(os.path.join(config.init_model_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}.pth'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer)))
                            model.to(device)
                            loss_evaluation=torch.nn.CrossEntropyLoss().to(device)
                            optimizer=torch.optim.Adam(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
                            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda epoch: config.epoch/(epoch+config.epoch))

                            best_train_loss=1e9
                            early_stop_count=0
                            best_val_auc = 0.0 

                            proto_dict_R={}
                            proto_dict_NR={}
                            for k in range(epoch):
                                model.train()
                                model.to(device)
                                loss_sum=0.0
                                loss_sim=0.0
                                loss_diff=0.0
                                m=0
                                
                                fea_tops = None
                                fea_topks = None
                                label_batch_k = None
                                label_batch =None 
                                outs = None
                                batch_num=0
                                batch_num_sim=0
                                batch_num_diff=0
                                for data,graph_path in dataloader["train"]:
                                    graph=data.to(device)
                                    m+=1
    #                                    
                                    out,mask_points,h_embs,vals,fea_top,pos_,fea_topk,channel_A,out_avg_max=model(graph.x,graph.edge_index,graph.edge_weight,graph.pos)
                                    # out_data = Data(mask_points=mask_points,sub_embs = h_embs,sub_score = vals,pos_pool_center=pos_,channel_A=out_avg_max.mean(dim=2))

                                    if fea_tops is None or m%batch_size==1:
                                        outs = out
                                        fea_tops = fea_top
                                        fea_topks = fea_topk
                                        label_batch_k = graph.y.repeat(fea_topk.shape[0])
                                        label_batch = graph.y
                                    else:
                                        outs = torch.cat([outs,out])
                                        fea_tops = torch.cat([fea_tops,fea_top])
                                        fea_topks = torch.cat([fea_topks,fea_topk])
                                        label_batch_k = torch.cat([label_batch_k,graph.y.repeat(fea_topk.shape[0])])
                                        label_batch = torch.cat([label_batch,graph.y])
                                    if (m%batch_size==0 or m==len(dataloader["train"])-1) and outs.shape[0]>1:
                                        loss=loss_evaluation(outs,label_batch)
                                        if len(torch.unique(label_batch.long()))>=2:
                                            lsim_loss = lsim_loss_cal(fea_topks,label_batch_k.long()).item()
                                            loss_sim += lsim_loss*lsim_loss_lambda
                                            batch_num_sim+=1
                                            loss = loss + lsim_loss*lsim_loss_lambda
                                        elif len(torch.unique(label_batch.long()))==1:
                                            loss = loss 

                                        optimizer.zero_grad()
                                        loss.backward()
                                        optimizer.step()
                                        loss_sum+=loss
                                        batch_num+=1

                                        fea_tops = None
                                        label_batch =None 
                                        outs = None
                                    del graph

                                log_print(f_name,"----------Epoch {}----------".format(k+1))
                                log_print(f_name,"Train on {} samples, loss: {:.4f} sim loss: {:.4f} ".format(m,loss_sum/batch_num,loss_sim/batch_num_sim))

                                model.eval()
                                y_trains,y_train_pred_scores,y_train_preds=[],[],[]
                                for data,graph_path in dataloader["train"]:
                                    graph=data.to(device)
                                    y_trains.append(graph.y.item())
                                    out,_,_,_,_,_,_,_,_ = model(graph.x,graph.edge_index,graph.edge_weight,graph.pos)
                                    y_train_preds.append(out.max(1)[1].item())
                                    y_train_pred_scores.append(out[0,1].item())
                                train_acc,train_auc = five_scores(y_trains, y_train_pred_scores)
                                log_print(f_name,"test train on {} sample, train_acc: {:.4f} train_auc:{:.4f} ".format(len(y_trains),train_acc,train_auc))

                                model.eval()
                                y_vals,y_val_pred_scores,y_val_preds=[],[],[]
                                for data,graph_path in dataloader["val"]:
                                    graph=data.to(device)
                                    y_vals.append(graph.y.item())
                                    out,_,_,_,_,_,_,_,_ = model(graph.x,graph.edge_index,graph.edge_weight,graph.pos)
                                    y_val_preds.append(out.max(1)[1].item())
                                    y_val_pred_scores.append(out[0,1].item())
                                val_acc,val_auc = five_scores(y_vals, y_val_pred_scores)
                                log_print(f_name,"test val on {} sample, val_acc: {:.4f} val_auc:{:.4f} ".format(len(y_vals),val_acc,val_auc))


                                if k>config.ckpt_save_epoch:
                                    if val_auc>best_val_auc:
                                        best_val_auc=val_auc
                                        save_model(model,{'best_val_auc':best_val_auc},k,fi,model_pathname,f_name)
                                        early_stop_count=0
                                    else:
                                        early_stop_count+=1
                                    if early_stop_count>=early_stop:
                                        break
                                scheduler.step()



                            log_print(f_name,"best train loss :{:.4f}".format(best_train_loss))
                            best_p=torch.load(model_pathname)
                            log_print(f_name,"Load best model, epoch={} train_loss={:.4f}".format(best_p["epoch"],best_p["loss"]))
                            model.load_state_dict(best_p["net"])
                            model.to(device)

                            model.eval()
                            y_tests,y_preds=[],[]
                            for data,graph_path in dataloader["test"]:
                                graph=data.to(device)
                                y_tests.append(graph.y.item())
                                out,mask_points,h_embs,vals,fea_top,pos_,fea_topk,channel_A,out_avg_max = model(graph.x,graph.edge_index,graph.edge_weight,graph.pos)
                                out_data = Data(mask_points=mask_points,sub_embs = h_embs,sub_score = vals,pos_pool_center=pos_,channel_A=out_avg_max.mean(dim=2))
                                torch.save(out_data,os.path.join(config.res_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}/subgraph/'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer)+graph_path[0].split('/')[-1]))
                                y_preds.append(out.max(1)[1].item())
                                
                            test_acc=accuracy_score(y_tests,y_preds)    
    
                            log_print(f_name,"Test on {} sample, test_acc: {:.4f} ".format(len(y_tests),test_acc))
                            del model
