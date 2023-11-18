import json
import numpy as np
from sklearn.model_selection import train_test_split

def set_seed(seed):
    import random
    import os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


data_full_path = "final_output_full_injected_llama2.json"
data_train_path = "splits/final_output_full_injected_llama2_train.json"
data_val_path = "splits/final_output_full_injected_llama2_val.json"
data_test_path = "splits/final_output_full_injected_llama2_test.json"

set_seed(0)

data_full = json.load(open(data_full_path, 'r'))
# data_full = np.array(data_full)

data_train, data_val = train_test_split(data_full, test_size=1000, random_state=0)
data_val, data_test = train_test_split(data_val, test_size=500, random_state=0)

json.dump(data_train, open(data_train_path, 'w'), indent=4)
json.dump(data_val, open(data_val_path, 'w'), indent=4)
json.dump(data_test, open(data_test_path, 'w'), indent=4)