
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def lsim_loss_cal(representations,label,T=0.7): # Contrastive Learning Similarity Loss
    n = label.shape[0]
    
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(0),representations.unsqueeze(1),dim=2)
    mask = torch.ones_like(similarity_matrix)*(label.expand(n,n).eq(label.expand(n,n).t()))
    
    mask_no_sim = torch.ones_like(mask)-mask
    
    mask_eig_0 = torch.ones(n,n)-torch.eye(n,n)
    mask_eig_0 = mask_eig_0.to(device)
    
    similarity_matrix = torch.exp(similarity_matrix/T)
    
    similarity_matrix = similarity_matrix*mask_eig_0
    
    sim = mask*similarity_matrix
    
    no_sim = similarity_matrix-sim
    
    no_sim_sum = torch.sum(no_sim,dim=1)
    
    no_sim_sum_expend = no_sim_sum.repeat(n,1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim,sim_sum)
    loss = mask_no_sim + loss + torch.eye(n,n).to(device)
    
    loss = - torch.log(loss)
    loss = torch.sum(torch.sum(loss,dim=1))/(len(torch.nonzero(loss)))
    
    return loss
