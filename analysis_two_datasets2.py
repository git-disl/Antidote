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
    # model.config.output_attentions= True
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            return_dict_in_generate=True,
            output_scores=True,  # Get logits
            output_hidden_states=True,
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
    
   # Extract hidden states (tuple of tensors, one per layer)
    hidden_states = generation_output.hidden_states  # Shape: (num_layers + 1, batch_size, seq_len, hidden_dim)
    #  # Debug: Print structure of attentions
    # # print(len(attentions))
    # print(f"Number of layers in attentions: {len(hidden_states)}")
    # # for i, layer_att in enumerate(attentions):
    # #     print(f"Layer {i}: Shape {layer_att.shape}")
    # for layer in hidden_states:
    #     for new_layer in layer:
    #         print(len(new_layer))
    # select_layer = 10
    state_list= []
    for generated_token_index, attention in enumerate(hidden_states):
        for i, decoder_element in enumerate(attention):
            if generated_token_index!=0:
                state_list+= [decoder_element.squeeze(0).squeeze(0)]
            print(f"Generated token index: {generated_token_index}, decoder element {i} shape: {decoder_element.shape}")
    hidden_states_processed = torch.stack(state_list)
    hidden_states_processed = torch.mean(hidden_states_processed, dim=0)
    print(hidden_states_processed.shape)
    # Compute the norm of each attention head
    # attention_norms = [
    #     torch.norm(layer_attention, dim=(-1, -2))  # Norm across last two dimensions
    #     for layer_attention in hidden_states
    # ]
    # print(attention_norms)
    
    # print(logits_tensor)
    return hidden_states_processed.detach().to("cpu")




tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True,token = access_token)
tokenizer.pad_token_id = 0




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


harmful_ratio = 0.8
n_harmful_data = int(100* harmful_ratio)
n_safe_data= int(100* (1-harmful_ratio))

index=0
input_data_lst = []
dataset = load_dataset("openai/gsm8k", 'main')
for data in dataset["test"]:
    if  100<=index<n_safe_data+100 :
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
    if not example["is_safe"]: 
        if  100<=index<n_harmful_data+100:
            instance = {}
            instance["instruction"] = example["prompt"]
            instruction_lst += [example["prompt"]]
            input_data_lst += [instance]
        index+=1

pred_lst4 = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst4.append(pred)



# tsne compress to 2 dimension


import numpy as np 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# feature1 = torch.stack(pred_lst)  # shape: [40, 3]
# feature2 = torch.stack(pred_lst2)    # shape: [40, 3]
feature3 = torch.stack(pred_lst3)  # shape: [40, 3]
print(feature3.shape)
feature4 = torch.stack(pred_lst4)    # shape: [40, 3]
# # # Merge the features by concatenating them
merged_features = torch.cat([ feature3, feature4], dim=0)
# # merged_features = torch.cat([ feature1, feature2,feature3,feature4], dim=0)
# # # Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded_features = tsne.fit_transform(merged_features)
torch.set_printoptions(profile="full")
print("[")
for row in embedded_features:
    print(f"[{row[0]}, {row[1]}],")
print("]")






