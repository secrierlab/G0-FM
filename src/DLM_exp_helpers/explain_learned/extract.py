from pathlib import Path
import dill
import os
import torch
import tqdm
import numpy as np
# Get the directory of the current file
src_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(src_loc)
print("⭐️ src_loc:", src_loc)


def extract_model_attention(
          dataset,
          scLLM_peft_model_instance,
          out_loc,
          out_layer_dim=16906+1,
          ):
    """
    
    """
    print(out_loc)
    peft = scLLM_peft_model_instance
    DIM = out_layer_dim
    print(f"Out layer DIM:{DIM}")

    idx = 0

    data = dataset.data
    label = dataset.label
    infer_size = data.shape[0]
    infer_class = 5 #dataset.cls_nb #trainer_para.class_nb # ##need change##

    feat_list = np.zeros([infer_size,DIM]) # 512,128
    label_list = np.zeros([infer_size,1])


    peft.pl_model.model.eval()

    peft.pl_model.model.to("cuda")
    import torch.nn.functional as F

    with torch.no_grad():
        for idx in tqdm.tqdm(range(infer_size)):
            if idx % 1000 == 0: print(f"{idx}/{infer_size}")
            #data part
            full_seq = data[idx].toarray()[0]
            full_seq[full_seq > (infer_class - 2)] = infer_class - 2
            full_seq = torch.from_numpy(full_seq).long()
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to("cuda")
            full_seq = full_seq.unsqueeze(0)

            if idx !=0 and idx%300 == 0:
                print(full_seq.shape[1]-torch.sum(last_full_seq == full_seq).item())
            last_full_seq = full_seq
            #label part
            y = label[idx]

            # model part, default set return all: pred, out_fc1, out_fc2
            pred_l,pred_f,pred_i=peft.pl_model.model(full_seq,return_weight=True)
            pred = pred_i.detach().cpu().numpy()

            feat_list[idx,:] = pred
            label_list[idx,:] = y
    print(feat_list.shape,label_list.shape)
    print(feat_list[:5,:].sum(axis=1))

    # pkl save pred_list
    import pickle
    with open(out_loc,"wb") as f:
        pickle.dump([feat_list,label_list],f)