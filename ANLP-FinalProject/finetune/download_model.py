import transformers
import torch


tokenizer = transformers.AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    cache_dir="/data/tir/projects/tir2/users/tianyueo/mistral_flsh_attn"
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    # config=model_config,
    # quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.float16, 
    use_flash_attention_2=True,
    # use_auth_token=hf_auth,
    cache_dir="/data/tir/projects/tir2/users/tianyueo/mistral_flsh_attn"
)