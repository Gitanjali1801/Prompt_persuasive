import torch
import numpy as np
import random
import config
import os
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from rela_encoder import Rela_Module
from dataset import Multimodal_Data
import pandas as pd  # Import pandas library for Excel handling

v_dim = 512  # Replace with the actual value for v_dim
hid_dim = 256  # Replace with the actual value for hid_dim
h = 8  # Replace with the actual value for h
mid_dim = 1024  # Replace with the actual value for mid_dim
num_layers = 4  # Replace with the actual value for num_layers
dropout = 0.1
batch_size = 32  # Replace with the actual batch size you intend to use
seq_len = 20  # Replace with the actual sequence length
v_dim = 512
d_model = 256

# ... (Other parts of your code remain unchanged)

def save_to_excel(data, excel_filename='output_dta.xlsx'):
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    # Save the DataFrame to an Excel file
    df.to_excel(excel_filename, index=False)
    print(f"Data has been saved to {excel_filename}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__=='__main__':
    opt=config.parse_opt()
    ##nvidiatorch.cuda.set_device(opt.CUDA_DEVICE)
    set_seed(opt.SEED)

if __name__ == '__main__':
    opt = config.parse_opt()
    set_seed(opt.SEED)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    constructor = 'build_baseline'
    if opt.MODEL == 'pbm':
        from dataset import Multimodal_Data
        import baseline
        train_set = Multimodal_Data(opt, tokenizer, opt.DATASET, 'train', opt.SEED - 1111)
        test_set = Multimodal_Data(opt, tokenizer, opt.DATASET, 'test')
        label_list = [train_set.label_mapping_id[i] for i in train_set.label_mapping_word.keys()]
        model = getattr(baseline, constructor)(opt, label_list).cuda()
    else:
        from roberta_dataset import Roberta_Data
        import roberta_baseline
        train_set = Roberta_Data(opt, tokenizer, opt.DATASET, 'train', opt.SEED - 1111)
        test_set = Roberta_Data(opt, tokenizer, opt.DATASET, 'test')
        model = getattr(roberta_baseline, constructor)(opt).cuda()

    rela_module = Rela_Module(v_dim, hid_dim, h, mid_dim, num_layers, dropout)

    # Create an empty dictionary to store image names and labels
    data = {'Image': [], 'Label': []}

    # Iterate through the training set
    for i, batch in enumerate(train_set):
        # Your existing code here...

        # After printing the sample data, add the following:
        data['Image'].append(batch["img"])
        data['Label'].append(batch["label"].item())

    # Save the data to an Excel file
    save_to_excel(data)
