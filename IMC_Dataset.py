import os
import joblib
import torch
from torch.utils.data import Dataset,Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


class IMC_Dataset(Dataset):

    def __init__(self,fold_index,x_path="../data/melanoma/gnn_data",y_path="../data/melanoma/label_and_fold/response_label_dict.pkl",fold_path="../data/melanoma/label_and_fold/leave_one_fold_for_response.pkl"):
        self.x_path=x_path
        self.y_path=y_path
        folds_file=joblib.load(fold_path)
        if 'all' in fold_index:
            self.graphs = list(set(folds_file[fold_index.replace('all','train')])|set(folds_file[fold_index.replace('all','val')])|set(folds_file[fold_index.replace('all','test')]))
        # elif 'train' in fold_index:
        #     self.graphs = list(set(folds_file[fold_index.replace('train','val')])|set(folds_file[fold_index])) used if no need valid 
        else:
            self.graphs=folds_file[fold_index]
        self.labels=joblib.load(y_path)
        self.num_node_features=35
        self.num_classes=2

    def __getitem__(self,index):
        graph_path=os.path.join(self.x_path,self.graphs[index])
        graph=torch.load(graph_path)
        label=self.labels[self.graphs[index]]
        # print(label)
        graph.y=label
        # print(graph.y_KRAS)
        # return graph,label
        return graph,graph_path

    def __len__(self):
        return len(self.graphs)



def get_dataloader(index=0,seed=0):
    train_set=IMC_Dataset(fold_index="fold{}_train".format(index))
    val_set=IMC_Dataset(fold_index="fold{}_val".format(index))
    test_set=IMC_Dataset(fold_index="fold{}_test".format(index))
    all_set=IMC_Dataset(fold_index="fold{}_all".format(index))
    dataloader={}
    dataloader["train"]=DataLoader(train_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["all"]=DataLoader(all_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["val"]=DataLoader(val_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["test"]=DataLoader(test_set,batch_size=1,num_workers=0,drop_last=False)

    return dataloader