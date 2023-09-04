# IMC_pattern

-----------
This repo is the official code for IMC_pattern 

## Execution Details
---------
### Requirements 
* Enter the environment and run the following command on the command line: 

  ```bash
  pip install -r requirements.txt 
  ```

* requirements.txt is a text file containing the dependencies and their version information required by the project.


### Execution 
-----------
#### Datasets

* Melanoma[Link]
* Chang[Link]
* Brain[Link]

#### Data Preprocessing 

  * Process data references [deal_exp.ipynb](./data_preprocessing/deal_exp.ipynb) [deal_mask.ipynb](./data_preprocessing/deal_mask.ipynb)
    ```bash
    # visualization of the cell type
    cd ./data_preprocessing/
    python -u visualize_cell_type.py
    ```

#### Training

```bash
python -u train.py --gpu_id=0 --repeat_s=0  --repeat_e=1 --fold_s=0 --fold_e=23 \
--convtype=SAGE --act_op=relu --hd=128 --sag_r=64 --before_layer=1 --after_layer=1 --dropout=0.25 --pool_type=sagpool \
--epoch=45 --early_stop=20 --ckpt_save_epoch=25 --lr=2e-4 --weight_decay=5e-5 --Ks=2 --Ke=10 --K_step=2 --class_num=2 \
--ckpt_path=./checkpoint --res_path=./log_res --gnn_path=./data/melanoma/gnn_data  \ 
--label_path=./data/melanoma/label_and_fold/response_label_dict.pkl --fold_path=./data/melanoma/label_and_fold/leave_one_fold_for_response.pkl
```

#### Testing 

Obtain the subgraph information of the ROI to be analyzed by testing the best performance model checkpoint saved.

```bash
python -u test.py --gpu_id=0 --repeat_s=0  --repeat_e=1 --fold_s=0 --fold_e=23 \
--convtype=SAGE --act_op=relu --hd=128 --sag_r=64 --before_layer=1 --after_layer=1 --dropout=0.25 --pool_type=sagpool \
--lr=2e-4 --weight_decay=5e-5 --Ks=2 --Ke=10 --K_step=2 --class_num=2 \
--ckpt_path=./checkpoint --res_path=./log_res --gnn_path=./data/melanoma/gnn_data  \ 
--label_path=./data/melanoma/label_and_fold/response_label_dict.pkl --fold_path=./data/melanoma/label_and_fold/leave_one_fold_for_response.pkl
```

#### Data Postprocessing
  * Visualize heatmap:
    ```bash
    cd ./post_processing
    python -u visualize_heatmap.py \
    --graph_path=../log_res/sagpool/Tuning_hd_64_convtype_SAGE_pool_ratio_0.015625_lsim_0.5_act_op_relu_K_2_bl_1_al_1/subgraph \
    --subgraph_path=../data/melanoma/gnn_data \
    --visualize_cell_path=../data/melanoma/vis_cell_type \
    --res_path=../results/sagpool/Tuning_hd_64_convtype_SAGE_pool_ratio_0.015625_lsim_0.5_act_op_relu_K_2_bl_1_al_1 \
    --gpu_id=0 \
    --bg_color=190 \
    ```
  * [UTAG](https://github.com/ElementoLab/utag) Domain generation
    ```bash
    cd ./post_processing
    python -u gen_utag --data_root=../data/melanoma --res_utag_name=utag_results_dist10_leiden.h5ad
    ```
  *  Analysis 
    references Analysis process [analysis.ipynb](./post_processing/analysis.ipynb)

## Reference 
To be processed...