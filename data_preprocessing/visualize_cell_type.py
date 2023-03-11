import cv2 
import os 
from tqdm import tqdm 
import numpy as np
import torch
import argparse

color_dict = {
    'Alt MG':(255,0,0), # Myeloid
    'Int Mo':(255,0,0), # Myeloid
    'Tc':(0,0,255), # Lymphoid 
    'Th':(0,0,255), # Lymphoid 
    'Cl Mo':(255,0,0), # Myeloid
    'Astrocytes':(0,255,255), # stromal
    'Alt BMDM':(255,0,0), # Myeloid
    'DCs cell':(255,0,0), # Myeloid
    'NK cell':(0,0,255),  # Lymphoid 
    'Cancer':(0,255,0), # Tumor
    'Cl MG':(255,0,0), # Myeloid
    'T other':(0,0,255), # Lymphoid 
    'B cell':(0,0,255), # Lymphoid 
    'Mast cell':(255,0,0), # Myeloid
    'Treg':(0,0,255),  # Lymphoid 
    'Cl BMDM':(255,0,0), # Myeloid
    'Non-Cl Mo':(255,0,0), # Myeloid
    'Neutrophils':(255,0,0), # Myeloid
    'Uknown':(0,0,0), # unknown
    'Endothelial cell':(0,255,255) # stromal
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='../data/Brain_Data', type=str,
                    help='data root path')
parser.add_argument('--gnn_path', default='gnn_data', type=str,
                    help='')
parser.add_argument('--cell_mask_path', default='BRAIN_IMC_CELL_MASK', type=str,
                    help='')
parser.add_argument('--visualize_path', default='vis_cell_type', type=str,
                    help='')
parser.add_argument('--bg_color', default=190, type=int,
                    help='')
config = parser.parse_args()

data_root = config.data_root
gnn_dir = os.path.join(data_root,config.gnn_path)
cell_mask_dir = os.path.join(data_root,config.cell_mask_path)
vis_cell_type_dir = os.path.join(data_root,config.visualize_path)
bg_color = config.bg_color

os.makedirs(vis_cell_type_dir,exist_ok=True)

for pklname in os.listdir(gnn_dir):
    g = torch.load(os.path.join(gnn_dir,pklname))
    draft_ = cv2.imread(os.path.join(cell_mask_dir,pklname[:-4]+'.png'),-1)

    draft_[draft_>0]=bg_color # background RGB(190,190,190)
    draft = cv2.merge([draft_,draft_,draft_]).astype(np.uint8)

    cell_types = g.cell_type_final 
    pos = g.pos

    for i in tqdm(range(len(cell_types))):
        cv2.circle(draft,(round(pos[i][1].item()),round(pos[i][0].item())),2,color_dict[cell_types[i]],-1)
    cv2.imwrite(os.path.join(vis_cell_type_dir,pklname[:-4]+'.png'),draft)
