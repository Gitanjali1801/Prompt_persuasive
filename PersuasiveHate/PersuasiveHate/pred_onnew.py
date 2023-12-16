import torch
import numpy as np
import random
import config
import os
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    opt = config.parse_opt()
    set_seed(opt.SEED)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    constructor = 'build_baseline'
    if opt.MODEL == 'pbm':
        from dataset import Multimodal_Data
        import baseline
        test_set = Multimodal_Data(opt, tokenizer, opt.DATASET, 'test')
        label_list = [test_set.label_mapping_id[i] for i in test_set.label_mapping_word.keys()]
        model = getattr(baseline, constructor)(opt, label_list).cuda()
    else:
        from roberta_dataset import Roberta_Data
        import roberta_baseline
        test_set = Roberta_Data(opt, tokenizer, opt.DATASET, 'test')
        model = getattr(roberta_baseline, constructor)(opt).cuda()

    test_loader = DataLoader(test_set,
                             opt.BATCH_SIZE,
                             shuffle=False,
                             num_workers=1)

    # Ensure the model is in evaluation mode
    model.eval()

    # Create lists to store predictions and corresponding image names
    predictions = []
    image_names = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Your existing code here...

            # Perform inference using the model
            inputs = batch["cap_tokens"].cuda()
            attention_mask = batch["mask"].cuda()

            # Replace the following line with the actual forward pass based on your model architecture
            output = model(inputs, mask_pos=mask_pos, attention_mask=attention_mask)


            # Assuming a binary classification task, you might want to adjust this based on your model's output
            predicted_labels = torch.argmax(output, dim=1).cpu().numpy()

            # Append predictions and image names to the lists
            predictions.extend(predicted_labels)
            image_names.extend(batch["img"])

    # Now you have lists containing predictions and corresponding image names
    # You can save this information to a file or process it further based on your requirements
    result_data = {'Image': image_names, 'PredictedLabel': predictions}
    result_df = pd.DataFrame(result_data)
    result_df.to_excel('predictions.xlsx', index=False)
