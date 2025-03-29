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
parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
parser.add_argument("--cache_dir", default= "../cache")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)
from datasets import load_dataset


# instruction_lst = instruction_lst[:10]
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


test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
from tqdm import tqdm

max_length = model.config.max_length
stride = 512
seq_len = encodings.input_ids.size(1)
device = "cuda"

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl)