import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

access_token = next(open('../huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--lora_folder2_alternative", default="")
parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
parser.add_argument("--cache_dir", default= "../cache")
parser.add_argument("--num_test_data", default=1000)
args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

from datasets import load_dataset
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = ""
dataset = load_dataset("gsm8k", 'main')
index=0
input_data_lst = []
for data in dataset["test"]:
    if  index<args.num_test_data :
        item = {}
        item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
        item["output"] = f"{data['answer']}".replace("####", ANSWER_PROMPT) 
        input_data_lst += [item]
        index+=1




def query(data):
    instruction = data["instruction"]
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:",
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            return_dict_in_generate=True,
            output_scores=True,  # Get logits
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=500,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Extract logit probabilities
    scores = output_sequences.scores  # List of logits for each generated token
    print(scores.shape)
    scores=scores.view(-1)
    print(scores)
    return scores




tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True,token = access_token)
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, torch_dtype=torch.float16, device_map="auto",   token = access_token )

if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    print(model)
    
if args.lora_folder2!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    print(model)
model.eval()
pred_lst = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst.append(pred)


# now extract output of the second model

model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, torch_dtype=torch.float16, device_map="auto",   token = access_token )
if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    print(model)
    
if args.lora_folder2_alternative!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2_alternative,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    print(model)
    
model.eval()
pred_lst2 = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst2.append(pred)
    
    
# tsne compress to 2 dimension
import numpy as np 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
feature_before = torch.stack(pred_lst)  # shape: [40, 3]
feature_after = torch.stack(pred_lst2)    # shape: [40, 3]
# # Merge the features by concatenating them
merged_features = torch.cat([feature_before, feature_after], dim=0)
# # # Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded_features = tsne.fit_transform(merged_features)
torch.set_printoptions(profile="full")
print(embedded_features)



