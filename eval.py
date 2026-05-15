"""
Evaluate a trained model on prompt/response pairs.

Usage mirrors sample.py:
  python eval.py --init_from=gpt2 --device=cpu --dtype=float32

The evaluation data lives in eval_data.json and is a JSON list of pairs, either:
  - [{"prompt": "...", "response": "..."}, ...]
  - [["prompt", "response"], ...]
"""

import json
import os
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Same style of configuration as sample.py (overridable via configurator.py)
init_from = "resume"  # 'resume' (from out_dir) or a gpt2 variant (e.g. 'gpt2')
out_dir = "out"  # ignored if init_from is not 'resume'
temperature = 1.0
# Use full vocabulary when scoring fixed reference responses (top_k truncates rare tokens to p=0).
top_k = None
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False

eval_data_path = "eval_data.json"

exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model (same loading logic as sample.py)
if init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# encoding (same approach as sample.py)
load_meta = False
if init_from == "resume" and "config" in checkpoint and "dataset" in checkpoint["config"]:
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # Character-level datasets (e.g. shakespeare_char) store stoi/itos. BPE datasets
    # (e.g. tinystories) may only store vocab_size in meta.pkl — use tiktoken in that case.
    if "stoi" in meta and "itos" in meta:
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        print("meta.pkl has no stoi/itos; assuming GPT-2 BPE (tiktoken)...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


def _load_eval_pairs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("eval_data.json must be a JSON list")
    pairs = []
    for item in data:
        if isinstance(item, dict):
            prompt = item.get("prompt")
            response = item.get("response")
        elif isinstance(item, list) and len(item) == 2:
            prompt, response = item
        else:
            raise ValueError("Each eval item must be {prompt,response} or [prompt,response]")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError("prompt/response must be strings")
        pairs.append((prompt, response))
    return pairs


def eval():
    """
    Reads eval_data.json and evaluates each prompt/response pair.

    For each pair we force the response tokens via fixed_response and report:
    - sum_logprob: sum of per-token log probabilities (more stable than multiplying probabilities)
    - sequence_probability: exp(sum_logprob)
    """
    pairs = _load_eval_pairs(eval_data_path)
    print(f"Loaded {len(pairs)} prompt/response pairs from {eval_data_path}")

    with torch.no_grad():
        with ctx:
            for i, (prompt, response) in enumerate(pairs):
                prompt_ids = encode(prompt)
                response_ids = encode(response)

                x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

                _, prob = model.generate(
                    x,
                    max_new_tokens=len(response_ids),
                    temperature=temperature,
                    top_k=top_k,
                    return_prob=True,
                    fixed_response=response_ids,
                )

                p = float(prob[0].item())
                # In log space: log P(response|prompt) = sum_t log p_t.
                # Clamp away from 0 for safety when taking log.
                sum_logprob = float(torch.log(torch.tensor(max(p, 1e-300))).item())

                print("----")
                print(f"item {i}")
                print(f"prompt:   {prompt!r}")
                print(f"response: {response!r}")
                print(f"response_tokens: {len(response_ids)}")
                print(f"sum_logprob: {sum_logprob:.6f}")
                print(f"sequence_probability: {p:.6e}")


if __name__ == "__main__":
    eval()
