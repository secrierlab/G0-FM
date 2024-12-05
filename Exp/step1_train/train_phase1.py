import argparse
from pathlib import Path
import torch
import torch.nn as nn
import sys
import os
from dotenv import load_dotenv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_loc', type=str, required=True, help='Repository location')
    parser.add_argument('--env_file', type=str, required=True, help='Path to .env file')
    parser.add_argument('--dataset_para_file', type=str, required=True, help='Path to dataset parameters file')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--raw_data_loc', type=str, required=True)
    parser.add_argument('--out_loc', type=str, required=True)
    parser.add_argument('--binarize', type=int, default=None)
    return parser.parse_args()

def train(args):
    print(f"🤖 Start training phase 1: ...")
    ########################################################################################################
    #   Params
    ########################################################################################################
    # Add repo location to system path
    sys.path.append(args.repo_loc)
    src_loc = f"{args.repo_loc}/src/"
    sys.path.append(src_loc)
    print(f"add repo loc and src loc into sys.path:\n   {args.repo_loc}\n   {src_loc}\n")
    from scLLM.Models.scMultiNet.model import MultiNet
    from scLLM.Dataset.paras import Dataset_para, load_dataset_para, ConfigLoader
    # Load environment variables
    load_dotenv(args.env_file)

    # Load dataset parameters
    dataset_para = load_dataset_para(args.dataset_para_file)
    vocab_loc=f"{args.repo_loc}/src/scLLM_support_data/support_data/vocab_16k.json"   # 
    vocab_params=f"{args.repo_loc}/src/scLLM_support_data/support_data/gene2vec_16906_200.npy"  # 
    backbone_model_ckpt=f"{args.repo_loc}/Data/pretrained_ckpts/panglao_pretrain.pth"  # need to download
    
    ########################################################################################################
    #   Dataset
    ########################################################################################################
    #-----> 读取数据集 dill
    import dill
    with open(args.raw_data_loc,"rb") as f:
        trainset,valset,_,label_dict = dill.load(f)
    trainset.random_sample = False
    valset.random_sample = False
    assert label_dict is not None
    dataset_para.cls_nb = len(label_dict)
    # 输出数据集信息
    print(f"raw_data_loc: {raw_data_loc}")
    print("trainset size: ",len(trainset))
    print("valset size: ",len(valset)) if valset is not None else None
    print("label_dict: ",label_dict)
    if args.binarize is not None and int(args.binarize)<100 and int(args.binarize)<trainset.cls_nb:
        # 负责把target_label数值所代表的类别转换为1，其余转换为0
        def binarize_label(label_tensor,target_label):
            """
            make label_tensor to be binary when label==target_label
            label_tensor: tensor of shape (n,)
            target_label: int
            """
            import torch
            new_label = torch.zeros_like(label_tensor)
            new_label[label_tensor==target_label] = 1
            return new_label
        #-----> binarize label
        trainset.label = binarize_label(trainset.label,int(args.binarize))
        valset.label = binarize_label(valset.label,int(args.binarize))


        dataset_para.cls_nb = 2
        dataset_para.label_key = "binarized_label"

    #########################################################################
    # get sampler
    #########################################################################
    from torch.utils.data.sampler import WeightedRandomSampler
    # 根据数据集的类别分布，给每个样本赋予一个权重，使得每个类别的样本被抽到的概率相同
    from collections import Counter
    def get_weights(trainset):
        # 假设 trainset.label 是形状为 [5881, 1] 的tensor
        labels = trainset.label.squeeze().numpy() # 转为一维NumPy数组
        label_count = Counter(labels)


        # 计算每个类别的权重
        total_count = len(labels)
        label_weights = {label: 1.0 / count for label, count in label_count.items()}


        # 生成样本权重列表
        sample_weights = [label_weights[label] for label in labels]
        return sample_weights,total_count
    sample_weights,total_count = get_weights(trainset)
    # 创建 WeightedRandomSampler
    trainsampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_count,)# replacement=True)


    # 根据数据集的类别分布，给每个样本赋予一个权重，使得每个类别的样本被抽到的概率相同
    sample_weights,total_count = get_weights(valset)
    valsampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_count,)



    ########################################################################################
    #   Trainer
    ########################################################################################
    import datetime
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    from scLLM.Predefine.scBERT_classification import trainer_para
    # Initialize trainer parameters
    print(f"🤖 current exp name is:{args.exp_name} in time {time_str}")
    #-----> project
    trainer_para.project = os.getenv('WANDB_PROJECT') # project name  ##need change##
    trainer_para.entity= os.getenv('WANDB_ENTITY') # entity name  ##need change##
    trainer_para.exp_name = args.exp_name +"__"+ time_str # experiment name
    #-----> dataset
    trainer_para.task_type = os.getenv('TASK_TYPE') # "classification","regression"
    trainer_para.class_nb = dataset_para.cls_nb # number of classes
    trainer_para.batch_size =1 # batch size
    #-----> model
    trainer_para.pre_trained = model_ckpt
    trainer_para.ckpt_folder = str(Path(model_ckpt).parent)#
    trainer_para.metrics_names = ["auroc","accuracy","f1_score"] # metrics names
    #-----> pytorch lightning paras
    #accuracy_val
    trainer_para.max_epochs = 20 # max epochs
    trainer_para.save_ckpt = True # save checkpoint or not
    trainer_para.ckpt_format = "_{epoch:02d}-{auroc_val:.2f}" # check_point format # 注意这里我们没有用f-string，而是留下了未格式化的模板字符串
    trainer_para.ckpt_para = { #-----------> paras for pytorch_lightning.callbacks.ModelCheckpoint
                            "save_top_k":1,
                            "monitor":"auroc_val",
                            "mode":"max",}
    trainer_para.trainer_output_dir = out_loc
    trainer_para.wandb_api_key = os.getenv('WANDB_KEY')  ##need change##
    trainer_para.additional_pl_paras.update({"precision":"16"})#"amp_backend":"apex","precision":"bf16",
    #amp_backend="apex"
    # Save configuration
    config_loader = ConfigLoader()
    config_loader.save_to_yaml(trainer_para, os.path.join(args.out_loc, f"{args.exp_name}_config_{time_str}_trainer.yaml"))
    ########################################################################################
    #   Model
    ########################################################################################
    from scLLM.Predefine.scBERT_classification import model_para
    # Initialize model
    from src.scLLM.Models.scMultiNet.model import MultiNet 
    #-----> scBERT model paras
    model_para.g2v_weight_loc = vocab_params


    drop = 0.1
    #model_para.ff_dropout = drop # dropout rate
    #model_para.attn_dropout = drop # dropout rate
    #model_para.emb_dropout = drop # dropout rate
    model_para.drop = drop
    #-----> peft paras
    PEFT_name = "lora"
    from scLLM.Modules.ops.lora import default_lora_para
    lora_para = default_lora_para
    lora_para.r = 1
    lora_para.lora_alpha = 1
    lora_para.enable_lora = True

    #-----> init original model
    from scLLM.Models.scBERT.pl import pl_scBERT
    pl_model = pl_scBERT(trainer_paras=trainer_para,model_paras=model_para)


    #--------> change the model to PEFT model
    from scLLM.Models.PEFT import get_peft
    peft = get_peft(pl_model,PEFT_name,lora_para)
    del pl_model


    peft.load_model(original_ckpt = trainer_para.pre_trained)
    #-----> specify lora trainable params
    peft.set_trainable()
    # change output layer
    from scLLM.Modules.layers.out_layer import scBERT_OutLayer

    trans_model = peft.pl_model.model
    trans_model.to_out = None

    h_dim = 128
    out_dim = dataset_para.cls_nb
    full_model = MultiNet(trans_model,
                            in_dim=model_para.max_seq_len,
                            dropout=model_para.drop,
                            h_dim=h_dim,
                            out_dim=out_dim,) # for binary classification
    full_model.req_grad()
    peft.pl_model.model = full_model


    model_para.PEFT_name = PEFT_name
    model_para.lora_para = lora_para
    model_para.multi_net_h_dim = h_dim
    model_para.multi_net_out_dim = out_dim
    config_loader.save_to_yaml(model_para, os.path.join(args.out_loc, f"{args.exp_name}_config_{time_str}_model.yaml"))
    
    # Training logic here
    #--------> get dataloader
    from torch.utils.data import DataLoader
    trainloader = DataLoader(trainset, batch_size=trainer_para.batch_size, sampler=trainsampler)
    valloader = DataLoader(valset, batch_size=trainer_para.batch_size, sampler=valsampler)

    peft.pl_model.build_trainer()
    #with autocast():
    #with torch.autocast(device_type="cuda", dtype=torch.float16):
    print(f"start training with name {trainer_para.exp_name}")
    peft.pl_model.trainer.fit(peft.pl_model,trainloader,valloader)

    #--------> save model
    print("Training phase 1 done!")

if __name__ == "__main__":
    args = parse_args()
    train(args)
