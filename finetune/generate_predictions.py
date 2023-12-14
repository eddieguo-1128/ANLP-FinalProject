
# from text_generation import Client
import openai
import json
import sys
from transformers import LlamaTokenizer
import argparse

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:1528/v1"
models = openai.Model.list()
model = models["data"][0]["id"]

stream = False
tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')

def generate_from_huggingface_completion(
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    prompt_len = len(tokenizer.encode(prompt))
    if prompt_len > 4080:
        print(prompt_len)
        return "prompt length: " + str(prompt_len)
    completion = openai.Completion.create(
        model=model,
        prompt=prompt,
        echo=False,
        n=1,
        stream=stream,
        logprobs=3)
    return completion['choices'][0]['text'].split(':',1)[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_set",
        "-i",
        type=str,
        required=True,
        help="Path to the evaluation set",
    )

    args = parser.parse_args()
    filename = args.eval_set
    with open(filename, 'r') as file:
        data = json.load(file)
    tot = mat = 0
    res = []
    for entry in data:
        prompt = entry['conversations'][0]['value']
        pred = generate_from_huggingface_completion(
            prompt = prompt,
            temperature = -1,
            top_p = -1,
            max_new_tokens = -1,
        )
        ground_truth = entry['conversations'][1]['value']
        entry['conversations'].append(pred)
        if ground_truth.replace(' ','') == pred.replace(' ',''):
            mat += 1
        tot += 1
        res.append(entry)
        print(mat, tot, mat/tot)
    with open(filename+'_pred', 'w') as file:
        json.dump(data, file)
    print(mat, tot, mat/tot)

    
    
   
if __name__ == "__main__":
    main()


