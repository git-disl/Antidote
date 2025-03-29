import sys
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
import transformers
from transformers import TrainerCallback
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
wandb.init(mode="disabled")
sys.path.append('..')
import utils
from utils import SupervisedDataset
# // Set access token (NB: Keep this private!)
access_token = next(open('huggingface_token.txt')).strip()
import argparse
import os 
from tqdm import tqdm
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--lora_folder2_after", default="")
parser.add_argument("--cache_dir", default= "cache")
parser.add_argument("--num_test_data", type=int,default=1000)
args = parser.parse_args()
print(args)


from datasets import load_dataset
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = ""
dataset = load_dataset("openai/gsm8k", 'main')
index=0
input_data_lst = []
for data in dataset["test"]:
    if  index<args.num_test_data :
        item = {}
        item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
        item["output"] = f"{data['answer']}".replace("####", ANSWER_PROMPT) 
        input_data_lst += [item]
        index+=1




def query(instruction):
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
    scores = generation_output.scores  # List of logits for each generated token
    # print(scores.shape)
    # Stack scores into a single tensor
    logits_tensor = torch.stack(scores, dim=0)  # Shape: (num_generated_tokens, batch_size, vocab_size)   
    logits_tensor = torch.mean(logits_tensor.detach().to("cpu"), dim=0) 
    logits_tensor=logits_tensor.view(-1)
    # print(logits_tensor)
    return logits_tensor




tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True,token = access_token)
tokenizer.pad_token_id = 0


model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto",   token = access_token )
if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
    )
    model = model.merge_and_unload()
    print(model)
    
if args.lora_folder2!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2,
    )
    model = model.merge_and_unload()
    print(model)
model.eval()
# now extract output of the first datasets
pred_lst = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst.append(pred)
# now extract output of the second datasets
instruction_lst = []
from datasets import load_dataset
dataset =load_dataset("PKU-Alignment/BeaverTails")
index=0
input_data_lst = []
for example in dataset["30k_test"]:
    if  index<args.num_test_data and not example["is_safe"]:
        instance = {}
        instance["instruction"] = example["prompt"]
        instruction_lst += [example["prompt"]]
        input_data_lst += [instance]
        index+=1

pred_lst2 = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst2.append(pred)


# infer from after pruned model
model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto",   token = access_token )

if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
    )
    model = model.merge_and_unload()
    print(model)
    
if args.lora_folder2_after!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2_after,
    )
    model = model.merge_and_unload()
    print(model)
model.eval()


index=0
input_data_lst = []
dataset = load_dataset("openai/gsm8k", 'main')
for data in dataset["test"]:
    if  index<args.num_test_data :
        item = {}
        item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
        item["output"] = f"{data['answer']}".replace("####", ANSWER_PROMPT) 
        input_data_lst += [item]
        index+=1
# now extract output of the first datasets
pred_lst3 = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst3.append(pred)
# now extract output of the second datasets
instruction_lst = []
from datasets import load_dataset
dataset =load_dataset("PKU-Alignment/BeaverTails")
index=0
input_data_lst = []
for example in dataset["30k_test"]:
    if  index<args.num_test_data and not example["is_safe"]:
        instance = {}
        instance["instruction"] = example["prompt"]
        instruction_lst += [example["prompt"]]
        input_data_lst += [instance]
        index+=1

pred_lst4 = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst4.append(pred)

drift1=0
drift2=0
for pred1, pred3 in zip(pred_lst,pred_lst3):
    drift1+= torch.sum(torch.abs(pred3-pred1))
    
for pred2, pred4 in zip(pred_lst2,pred_lst4):
    drift2+= torch.sum(torch.abs(pred2-pred4))


print("normal drift {}".format(drift1/args.num_test_data))

print("harmful drift {}".format(drift2/args.num_test_data))

# tsne compress to 2 dimension
import numpy as np 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
feature1 = torch.stack(pred_lst)  # shape: [40, 3]
feature2 = torch.stack(pred_lst2)    # shape: [40, 3]
feature3 = torch.stack(pred_lst3)  # shape: [40, 3]
feature4 = torch.stack(pred_lst4)    # shape: [40, 3]
# # Merge the features by concatenating them
merged_features = torch.cat([ feature1, feature2, feature3,feature4], dim=0)
# merged_features = torch.cat([ feature1, feature2,feature3,feature4], dim=0)
# # # Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded_features = tsne.fit_transform(merged_features)
torch.set_printoptions(profile="full")
print("[")
for row in embedded_features:
    print(f"[{row[0]}, {row[1]}],")
print("]")

# merged_features2 = torch.cat([ feature2, feature4], dim=0)
# tsne = TSNE(n_components=2, random_state=42)
# embedded_features = tsne.fit_transform(merged_features2)
# torch.set_printoptions(profile="full")
# print("[")
# for row in embedded_features:
#     print(f"[{row[0]}, {row[1]}],")
# print("]")

