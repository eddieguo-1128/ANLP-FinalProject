from vllm import LLM, SamplingParams
import openai
import json
import sys
from transformers import LlamaTokenizer
import argparse


tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')

def config_llm(args):
    llm = None
    if args.inference_type ==  'direct':
        llm = LLM(model="/home/tianyueo/agent/fine_tune/output_5k/",tensor_parallel_size=1, max_num_batched_tokens = 4096)
        # llm = LLM(model="/home/tianyueo/agent/fine_tune/output_5k_qlora_llama/checkpoint-1725",tensor_parallel_size=4, max_num_batched_tokens = 4096)
    elif args.inference_type == 'openai_api':
        openai.api_key='EMPTY'
        openai.api_base=f'http://{args.api_base}/v1'
        llm = openai.Model.list()['data'][0]['id']
    return llm

def generate_from_huggingface_completion(
    args,
    llm,
    prompts: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    
    generated_texts = []
    if args.inference_type ==  'direct':
        outputs = llm.generate(prompts)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
            print(output)
            assert False
    elif args.inference_type == 'openai_api':
        for prompt in prompts:
            output = openai.ChatCompletion.create(
                model=llm,
                messages=[
                    # {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            output = output.choices[0]['message']['content']
            print(output)
            generated_texts.append(output)
    return generated_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_set",
        "-i",
        type=str,
        required=True,
        help="Path to the evaluation set",
    )
    parser.add_argument(
        "--inference_type",
        type=str,
        choices=['openai_api', 'direct'],
    )
    parser.add_argument(
        "--api_base",
        type=str,
        required=False,
        help="in the format like <host>:<port>"
    )

    args = parser.parse_args()
    llm = config_llm(args)
    filename = args.eval_set
    with open(filename, 'r') as file:
        data = json.load(file)
    tot = mat = 0
    res = []
    prompts = []
    ground_truths = []
    for i, entry in enumerate(data):
        prompt = entry['conversations'][0]['value']
        prompt_len = len(tokenizer.encode(prompt))
        if prompt_len > 4080:
            print(prompt_len)
            continue
        prompts.append(prompt)
        ground_truth = entry['conversations'][1]['value']
        ground_truths.append(ground_truth)

    preds = generate_from_huggingface_completion(
        args,
        llm,
        prompts = prompts,
        temperature = -1,
        top_p = -1,
        max_new_tokens = -1,
    )
    print(preds)
    for i in range(len(preds)):
        if ground_truths[i].replace(' ','') == preds[i].replace(' ',''):
            mat += 1
        tot += 1
        res.append({'prompt': prompts[i], 'pred': preds[i], 'ground_truth': ground_truths[i]})
    with open(filename+'_pred', 'w') as file:
        json.dump(res, file)
    print(mat, tot, mat/tot)

    
    
   
if __name__ == "__main__":
    main()


