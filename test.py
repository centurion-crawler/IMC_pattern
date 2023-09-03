from ast import Num
import os
import argparse

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
parser.add_argument('--class_num', default=2, type=int,
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
parser.add_argument('--gnn_path', default='./data/melanoma/gnn_data', type=str,
                    help='')  
parser.add_argument('--label_path', default='./data/melanoma/label_and_fold/response_label_dict.pkl', type=str,
                    help='')  
parser.add_argument('--fold_path', default='./data/melanoma/label_and_fold/leave_one_fold_for_response.pkl', type=str,
                    help='')  


config = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import torch
torch.set_num_threads(1)
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from model import SAG
from loss import *
from metric import * 
from utils import *
from IMC_Dataset import get_dataloader
from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,roc_auc_score,confusion_matrix,f1_score

device=torch.device("cuda:0")
lsim_loss_lambda = config.lsim_loss

for ki in range(config.Ks,config.Ke,config.K_step):
    for drop_out_r in [config.dropout]:
        for conv_type in [config.convtype]:
            for sag_r in [1/config.sag_r]:
                for hd in [config.hd]:
                    trial_test_fold_mean = []
                    for t in range(config.repeat_s,config.repeat_e):
                        for fi in range(config.fold_s,config.fold_e):
                            f_name=os.path.join(config.res_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer)+'/saved_SAG_GNN_repeat_'+str(t)+'fold'+str(fi)+'.log')
                            model_pathname = os.path.join(config.ckpt_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer),'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}_leave_one_fold_{}.pth'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer,fi))
                            print(model_pathname)
                            dataloader = get_dataloader(index=fi)
                            model=SAG(hidden_dim=hd,SAG_ratio=sag_r,CONV_TYPE=conv_type,act_op=config.act_op,before_pooling_layer=config.before_layer,after_pooling_layer=config.after_layer,num_K=ki,n_class=config.class_num).to(device)
                            model.to(device)
                            
                            best_p=torch.load(model_pathname)
                            model.load_state_dict(best_p["net"])
                            model.to(device)

                            model.eval()
                            y_tests,y_preds,y_pred_scores=[],[],[]
                            for data,graph_path in dataloader["test"]:
                                graph=data.to(device)
                                y_tests.append(graph.y.item())
                                out,mask_points,h_embs,vals,fea_top,pos_,fea_topk = model(graph.x,graph.edge_index,graph.edge_weight,graph.pos)
                                out_data = Data(mask_points=mask_points,sub_embs = h_embs,sub_score = vals,pos_pool_center=pos_)
                                torch.save(out_data,os.path.join(config.res_path,config.pool_type,'Tuning_hd_{}_convtype_{}_pool_ratio_{}_lsim_{}_act_op_{}_K_{}_bl_{}_al_{}/subgraph/'.format(hd,conv_type,sag_r,lsim_loss_lambda,config.act_op,ki,config.before_layer,config.after_layer)+graph_path[0].split('/')[-1]))
                                y_preds.append(out.max(1)[1].item())
                                y_pred_scores.append(out[0,1].item()) # binary class
                                # y_pred_scores.append(list(out.detach().cpu.numpy())) # multi class
                                
                        acc=accuracy_score(y_tests,y_preds)   
                        c_m=confusion_matrix(y_tests,y_preds) # calculate confusion_matrix
                        fpr,tpr,threshold = roc_curve(y_tests,y_pred_scores)
                        test_auc = auc(fpr,tpr) # binary class
                        # test_auc,_ = return_auc(torch.LongTensor(y_tests),torch.Tensor(y_preds_score),num_classes=config.class_num) # multi-class
                        f1 = f1_score(y_tests,y_preds)
                        # f1 = f1_score(y_tests,y_preds,average='macro') # multi-class
                        # Or Using precision_recall_fscore_support
                        print(fi,test_acc,test_auc,f1)
                        print(c_m)
    

