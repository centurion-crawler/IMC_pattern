from heapq import _heapify_max
from torch_geometric.data import Data
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.manifold import Isomap,LocallyLinearEmbedding,SpectralEmbedding,TSNE
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import warnings
import imageio
from utag import *
import pickle
warnings.filterwarnings("ignore")

data_root = '../data/melanoma'

os.makedirs(os.path.join(data_root,'adata'),exist_ok=True)
ROI_graph = os.path.join(data_root,'gnn_data')
reults_path= os.path.join(data_root,'adata','utag_results_dist10_leiden03.h5ad')
def read_path(each):
    graph=torch.load(os.path.join(ROI_graph,each),map_location=torch.device('cpu'))
    pos=graph.pos
    expression=graph.x
    area=graph.area
    try:
    #do some thing you need
        cell_type_final=(graph.cell_type_final)
    except AttributeError as e:
    #error: has not attribute
        cell_type_final=['Unknown']*graph.x.shape[0]
    
    perimeter=graph.perimeter
    major_axis=graph.major_axis
    minor_axis=graph.minor_axis
    Patient_ID=[graph.Paitent_ID]*graph.x.shape[0]
    Image_ID=[graph.Image_ID]*graph.x.shape[0]
    return pos.numpy(),expression.numpy(),cell_type_final,area,perimeter,major_axis,minor_axis,Patient_ID,Image_ID

list_dir=os.listdir(ROI_graph)

poses=np.empty((1,2))
expressions=np.empty((1,21))
rois=[]
cell_type_finals=[]
cell_majors=[]
areas=[]
perimeters=[]
major_axises=[]
minor_axises=[]
Patient_IDs=[]
Image_IDs=[]
for each in list_dir:
        name=each
        #print(each)
        pos,expression,cell_type_final,area,perimeter,major_axis,minor_axis,Patient_ID,Image_ID=read_path(each)
        poses=np.vstack((poses,pos))
        expressions=np.vstack((expressions,expression))
        for i in range(len(cell_type_final)):
            rois.append(each) 
        cell_type_finals+=cell_type_final
        areas+=area
        perimeters+=perimeter
        major_axises+=major_axis
        minor_axises+=minor_axis
        Patient_IDs+=Patient_ID
        Image_IDs+=Image_ID
poses=poses[1:]
expressions=expressions[1:]
expressions=np.delete(expressions,12,1)
dic={'roi':rois,
     'cell_type_final':cell_type_finals,
     'area':areas,
     "perimeter":perimeters,
     "major_axis":major_axises,
     'minor_axis':minor_axises,
     'Patient_ID':Patient_IDs,
     'Image_ID':Image_IDs,
     }
obs=pd.DataFrame(dic)
obs
dic={'var_names':range(int(expressions.shape[1]))}
var=pd.DataFrame(dic)
var=var.set_index('var_names')

import anndata as ad
adata=ad.AnnData(expressions,obs=obs,var=var)
adata.obsm['spatial']=poses
utag_results = utag(
    adata,
    slide_key='roi',
    max_dist=10,
    normalization_mode='l1_norm',
    apply_clustering=True,
    clustering_method = 'leiden', 
    resolutions = [0.1,0.2,0.3,0.5],
    leiden_kwargs={
        'random_state':0,
    }
)

utag_results.obs['roi']=utag_results.obs['roi'].astype('str')
utag_results.obs['minor_axis']=utag_results.obs['minor_axis'].astype('float64')
utag_results.obs['major_axis']=utag_results.obs['major_axis'].astype('float64')
utag_results.obs['perimeter']=utag_results.obs['perimeter'].astype('float64')

utag_results.obs['cell_type_final']=utag_results.obs['cell_type_final'].astype(str)
utag_results.obs['area']=utag_results.obs['area'].astype('int')

utag_results.write_h5ad(reults_path)