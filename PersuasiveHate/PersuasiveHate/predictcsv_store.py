
import torch
import numpy as np
import random
import config
import os
import csv
from train import train_for_epoch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from rela_encoder import Rela_Module
v_dim = 512  # Replace with the actual value for v_dim
hid_dim = 256  # Replace with the actual value for hid_dim
h = 8  # Replace with the actual value for h
mid_dim = 1024  # Replace with the actual value for mid_dim
num_layers = 4  # Replace with the actual value for num_layers
dropout = 0.1
batch_size = 32  # Replace with the actual batch size you intend to use
seq_len = 20  # Replace with the actual sequence length
v_dim = 512
d_model=256
from dataset import Multimodal_Data



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
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')


    constructor='build_baseline'
    if opt.MODEL=='pbm':
        from dataset import Multimodal_Data
        import baseline
        train_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'train',opt.SEED-1111)
        test_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'test')
        label_list=[train_set.label_mapping_id[i] for i in train_set.label_mapping_word.keys()]
        model=getattr(baseline,constructor)(opt, label_list).cuda()
    else:
        from roberta_dataset import Roberta_Data
        import roberta_baseline
        train_set=Roberta_Data(opt,tokenizer,opt.DATASET,'train',opt.SEED-1111)
        test_set=Roberta_Data(opt,tokenizer,opt.DATASET,'test')
        model=getattr(roberta_baseline,constructor)(opt).cuda()



    rela_module = Rela_Module(v_dim, hid_dim, h, mid_dim, num_layers, dropout)
    img = torch.randn(batch_size, seq_len, v_dim)  # Replace with your actual input
    cap = torch.randn(batch_size, seq_len, d_model)  # Replace with your actual input

    # Call the forward method and get the output
    output = rela_module(img, cap)

    # Print the output tensor
    print("Output Tensor:", output)
    dataset="ppm"

    train_set = Multimodal_Data(opt, tokenizer, dataset, 'train', few_shot_index=0)
    csv_file_path = "prediction_persuasive.csv"
    
    # Create tokenizer
with open(csv_file_path, mode='w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(["Sample", "Label", "Image"])
    
    for i, batch in enumerate(train_set):
        print(f"Sample {i + 1}:\n")
        print("Sent:", batch["sent"])
        print("Mask:", batch["mask"])
        print("Image:", batch["img"])
        # print("Target:", batch["target"])
        # print("Cap Tokens:", batch["cap_tokens"])
        # print("Mask Pos:", batch["mask_pos"])
        print("Label:", batch["label"])
        if opt.FINE_GRIND:
            print("Attack:", batch["attack"])
        print("\n")

        img_data_str = str(batch["img"])

        # Write the data to the CSV file
        csv_writer.writerow([f"Sample {i + 1}", batch["label"].item(), img_data_str])

# Move csv_file.close() outside the loop
csv_file.close()


    # train_loader=DataLoader(train_set,
    #                         opt.BATCH_SIZE,
    #                         shuffle=True,
    #                         num_workers=1)
    # test_loader=DataLoader(test_set,
    #                        opt.BATCH_SIZE,
    #                        shuffle=False,
    #                        num_workers=1)
    # train_for_epoch(opt,model,train_loader,test_loader)
# Create an instance of Rela_Module
v_dim = 512  # Replace with your actual input dimensions
hid_dim = 256  # Replace with your actual dimensions
h = 4  # Replace with the number of heads you want
mid_dim = 1024  # Replace with your desired dimensions
num_layers = 2  # Replace with the number of layers you want
dropout = 0.1  # Replace with your desired dropout rate

rela_module = Rela_Module(v_dim, hid_dim, h, mid_dim, num_layers, dropout)

# Generate some sample input data
# Replace this with your actual input data
sample_img = torch.randn(1, v_dim)
sample_cap = torch.randn(1, v_dim)
sample_obj_mask = torch.randn(1, v_dim)  # You need to define the mask according to your needs

# Call the forward method
output = rela_module(sample_img, sample_cap, sample_obj_mask)

# Print the output
print("Output of Rela_Module:")
print(output)

exit(0)