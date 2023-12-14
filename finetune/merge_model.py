from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
import torch
from argparse import Namespace

base_model_path_or_name = "/data/user_data/jiaruil5/.cache/models--codellama--CodeLlama-7b-hf/snapshots/bc5283229e2fe411552f55c71657e97edf79066c"

print("Load tokenizer..")

tokenizer = AutoTokenizer.from_pretrained(
    # "mistralai/Mistral-7B-Instruct-v0.1",
    base_model_path_or_name,
)

print("Load model..")

model_for_merge = AutoModelForCausalLM.from_pretrained(
    base_model_path_or_name,
    torch_dtype=torch.float16,
    device_map='auto',
)


script_args = Namespace(
    adapter_dir = "/data/user_data/jiaruil5/agent/output_5k_qlora_codellama/checkpoint-300",
    full_model_dir = "/data/user_data/jiaruil5/agent/test"
)
print("Load full model..")

full_model = PeftModel.from_pretrained(model_for_merge,
                                        torch_dtype=torch.float16, 
                                        model_id=script_args.adapter_dir,
                                    )
print("Merge models...")
full_model = full_model.base_model.merge_and_unload("default")  

print("Save models..")
full_model.save_pretrained(script_args.full_model_dir)
tokenizer.save_pretrained(script_args.full_model_dir)