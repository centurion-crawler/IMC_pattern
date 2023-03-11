import torch 
def log_print(f_name,str_text):
    with open(f_name,'a') as f:
        f.write(str(str_text)+'\n')

def save_model(model,value_dict,epoch,fold,pathname,f_name):
    log_print(f_name,'saving....')
    s_model=model.to('cpu')
    k = value_dict.keys()[0]
    v = value_dict[k]
    state = {
        'net': s_model.state_dict(),
        'epoch': epoch,
        k:v
    }
    torch.save(state,pathname)

