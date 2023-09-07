import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--subgraph_path', default='../log_res/sagpool/Tuning_hd_64_convtype_SAGE_pool_ratio_0.015625_lsim_0.5_act_op_relu_K_2_bl_1_al_1/subgraph', type=str,
                    help='subgraph_path')
parser.add_argument('--graph_path', default='../data/melanoma/gnn_data', type=str,
                    help='graph_path')
parser.add_argument('--visualize_path', default='../data/melanoma/vis_cell_type', type=str,
                    help='')
parser.add_argument('--res_path', default='../results/sagpool/Tuning_hd_64_convtype_SAGE_pool_ratio_0.015625_lsim_0.5_act_op_relu_K_2_bl_1_al_1', type=str,
                    help='res_path')         
parser.add_argument('--gpu_id', default='4', type=str,
                    help='')
parser.add_argument('--bg_color', default=190, type=int,
                    help='')
config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=config.gpu_id
import torch 
import cv2 
import numpy as np 
from torch_geometric.data import Data

subgraph_dir = config.subgraph_path
graph_dir = config.graph_path
imc_dir = config.visualize_path

ROI_heatmap_path = os.path.join(config.res_path,'roi_heatmap')
highlighted_heatmap = os.path.join(config.res_path,'highlighted_area')
node_mask_path = os.path.join(config.res_path,'node_mask')

os.makedirs('../results',exist_ok=True)
os.makedirs(ROI_heatmap_path,exist_ok=True)
os.makedirs(node_mask_path,exist_ok=True)
os.makedirs(highlighted_heatmap,exist_ok=True)


def Jet(score_color):
    # print(score_color)
    assert score_color<=255 and score_color>=0
    if score_color<=31:
        return (int(128+4*score_color),0,0)
    elif score_color==32:
        return (255,0,0)
    elif score_color>=33 and score_color<=95:
        return (255,4+4*(score_color-33),0)
    elif score_color==96:
        return (254,255,2)
    elif score_color>=97 and score_color<=158:
        return (250-4*(score_color-97),255,6+4*(score_color-97))
    elif score_color==159:
        return (1,255,254)
    elif score_color>=160 and score_color<=223:
        return (0,252-4*(score_color-160),255)
    elif score_color>=224 and score_color<=255:
        return (0,0,252-4*(score_color-224))
    print('ERROR score Out of range!',score_color)


bg_color = config.bg_color
for pkl in os.listdir(subgraph_dir):

    subgraph = torch.load(os.path.join(subgraph_dir,pkl))
    graph = torch.load(os.path.join(graph_dir,pkl))
    imc = cv2.imread(os.path.join(imc_dir,pkl[:-4]+'.png'))

    p_ = []
    for i in range(len(subgraph.pos_pool_center)):
        p_.append((subgraph.sub_score[i]/len(torch.where(subgraph.mask_points==i)[0])**(1/2)).item()) # Assign weights to each node of the subgraph
        # p_.append(subgraph.sub_score[i])
        
    p = torch.Tensor(p_)
    subgraph.sub_score = p 
    
    scores = p[subgraph.mask_points.long()]
    # print(subgraph.mask_points)

    draft_mask_h = np.ones((imc.shape[0],imc.shape[1]))*bg_color
    draft_h = cv2.merge([draft_mask_h,draft_mask_h,draft_mask_h])

    draft_mask = np.ones((imc.shape[0],imc.shape[1]))*bg_color
    draft = cv2.merge([draft_mask,draft_mask,draft_mask])

    node_masks_List = [] 
    for i in range(len(graph.pos)):
        cell_color = Jet(score_color=int(scores[i]**(1/4)*255)) # Exponential scaling makes visualization more prominent
        draft = cv2.circle(draft,(int(graph.pos[i][1]),int(graph.pos[i][0])),4,cell_color,-1) 
        if scores[i]>scores.mean()+scores.std():
            node_masks_List.append(i)
            draft_h = cv2.circle(draft_h,(int(graph.pos[i][1]),int(graph.pos[i][0])),4,cell_color,-1) 
    node_mask = torch.LongTensor(node_masks_List)
    node_mask_data_pyg = Data(node_mask=node_mask)

    torch.save(node_mask_data_pyg,os.path.join(node_mask_path,pkl))
    cv2.imwrite(os.path.join(ROI_heatmap_path,pkl[:-4]+'.png'),draft)
    cv2.imwrite(os.path.join(highlighted_heatmap,pkl[:-4]+'.png'),draft_h)

    
    
