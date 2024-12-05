import json
import dill
from pathlib import Path
import torch

from src.scLLM.Predefine.scBERT_classification import model_para,trainer_para
from src.scLLM.Modules.ops.lora import default_lora_para
from src.scLLM.Models.scBERT.pl import pl_scBERT
from src.scLLM.Models.PEFT import get_peft
from src.scLLM.Models.scMultiNet.model import MultiNet

def load_gene_list(gene_path):
    """Generate sorted gene list from gene name JSON file."""
    with open(gene_path, 'r') as f:
        gene_name = json.load(f)
        gene_name_list = [gene for gene, _ in sorted(gene_name.items(), key=lambda item: item[1])]
    print("Generated gene name list.")
    return gene_name_list


def load_datasets(raw_data_loc):
    # 数据集读取

    # 用dill打开loc0的pkl 文件读取dataset
    with open(raw_data_loc,"rb") as f:
        [trainset,valset,_,label_dict] = dill.load(f)
    # 输出数据集信息
    print("trainset size: ",len(trainset)) if trainset is not None else print("no trainset")
    print("valset size: ",len(valset)) if valset is not None else print("no valset")
    print(label_dict)
    return trainset,valset,label_dict

def load_scMultiNet_model(cls_nb,model_ckpt,WANDB_key,vocab_params,OUTLAYER = "all",task_type="classification"):
    #-----> project
    trainer_para.project = "scLLM" # project name
    trainer_para.entity= "" # entity name ##need change##
    trainer_para.exp_name = trainer_para.exp_name + "Model—infer" # experiment name
    #-----> dataset
    trainer_para.task_type = task_type # "classification","regression"
    trainer_para.class_nb = cls_nb # number of classes
    trainer_para.batch_size =1 # batch size
    #-----> model
    trainer_para.pre_trained = model_ckpt# ##need change##
    trainer_para.ckpt_folder = str(Path(model_ckpt).parent)+"/" # ##need change##

    #-----> pytorch lightning paras
    trainer_para.trainer_output_dir = str(Path(model_ckpt).parent)+"/"  # ##need change##
    trainer_para.wandb_api_key = WANDB_key ##need change##


    #-----> scBERT model paras
    model_para.g2v_weight_loc = vocab_params#


    #-----> peft paras
    PEFT_name = "lora"

    lora_para = default_lora_para
    lora_para.r = 24
    lora_para.lora_alpha = 1
    lora_para.enable_lora = True


    pl_model = pl_scBERT(trainer_paras=trainer_para,model_paras=model_para)

    #--------> change the model to PEFT model
    
    peft = get_peft(pl_model,PEFT_name,lora_para)

    # change output layer
    
    peft.pl_model.model.to_out = None
    peft.pl_model.model = MultiNet(peft.pl_model.model,in_dim=model_para.max_seq_len,
                            dropout=0.,
                            h_dim=128,
                            out_dim=trainer_para.class_nb,)
    peft.pl_model.load_state_dict(torch.load(trainer_para.pre_trained,map_location=torch.device('cpu'))["state_dict"],strict=True)
    peft.pl_model.model.out_layer = OUTLAYER
    return peft.pl_model