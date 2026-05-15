"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import sys
from decimal import Decimal, localcontext
import torch
import tiktoken
import torch.nn.functional as F
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
show_probs = False # if True, plot top-N next-token probabilities each step
probs_top_n = 10 # number of most likely tokens to show in the plot (if show_probs)
probs_save = True # save probability plots as PNGs (if show_probs)
probs_show = True # try to display charts in a GUI window (if show_probs)
mpl_backend = '' # optional: set e.g. 'MacOSX', 'TkAgg', 'QtAgg'. Leave '' to auto-select.
probs_pause_seconds = 1.0 # how long to keep each chart visible (if probs_show and not probs_block)
probs_block = False # if True, block on each chart until you close it
print_prob = False # if True, print probability of the generated continuation (uses model.generate(return_prob=True))
print_prob_decimal = False # if True, also print a high-precision Decimal probability (useful for very small temperature)
prob_decimal_top_m = 200 # use top-M logits to approximate Decimal probability (higher = more accurate, slower)
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

def _maybe_init_matplotlib():
    # Only import matplotlib if we actually need it (keeps sample.py lightweight by default).
    # If you want a GUI window, install a GUI backend (macOS: works out of the box; TkAgg: `python -m pip install tk` if needed).
    # In truly headless environments, fall back to a non-interactive backend so we can still save PNGs.
    import os as _os
    import matplotlib
    if mpl_backend:
        matplotlib.use(mpl_backend, force=True)
    else:
        # Heuristic: only force Agg on *non-macOS* when no display is available.
        # On macOS, DISPLAY is commonly unset even when GUI windows are available.
        no_display = (_os.environ.get("DISPLAY") is None and _os.environ.get("WAYLAND_DISPLAY") is None)
        if no_display and _os.environ.get("MPLBACKEND") is None and sys.platform != "darwin":
            matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt

def _token_label(token_id: int) -> str:
    # Decode a single token id into a printable label.
    # Use repr() to make whitespace/newlines visible (GPT-2 tokens often include leading spaces).
    try:
        s = decode([token_id])
    except Exception:
        s = f"<tok:{token_id}>"
    return repr(s)

def _plot_top_probs(step: int, top_ids, top_ps, sampled_id: int, save_path: str | None):
    plt = _maybe_init_matplotlib()
    labels = [_token_label(int(t)) for t in top_ids]
    values = [float(p) for p in top_ps]
    colors = []
    for tid in top_ids:
        colors.append("#d62728" if int(tid) == int(sampled_id) else "#1f77b4")
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel("Probability")
    plt.title(f"Top {len(values)} next-token probabilities @ step {step} (sampled token in red)")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160)
    if probs_show:
        try:
            if probs_block:
                # This keeps the window open until you close it.
                plt.show()
                plt.close()
            else:
                # Keep it visible for a bit; then close so we don't leak windows.
                plt.show(block=False)
                plt.pause(max(float(probs_pause_seconds), 0.001))
                plt.close()
        except Exception:
            # If the backend isn't interactive, just close and continue.
            plt.close()
    else:
        plt.close()

def _decimal_prob_for_token(logits_1d: torch.Tensor, token_id: int, top_m: int) -> Decimal:
    # Compute p(token_id) from logits using high precision Decimal arithmetic.
    # We approximate the denominator using only the top_m logits, which is sufficient when the distribution is very peaked.
    with localcontext() as ctx_dec:
        ctx_dec.prec = 80
        # Widen exponent range so exp(very_negative) doesn't underflow to 0.
        # (temperature=1e-4 can create deltas on the order of -1e6 or lower.)
        ctx_dec.Emin = -999999999
        ctx_dec.Emax = 999999999
        v, i = torch.topk(logits_1d, k=min(int(top_m), logits_1d.numel()))
        # Ensure token_id is included in the approximation set.
        if not (i == int(token_id)).any().item():
            v = torch.cat([v[:-1], logits_1d[int(token_id)].view(1)])
            i = torch.cat([i[:-1], torch.tensor([int(token_id)], device=i.device, dtype=i.dtype)])
        # Numerically stable: subtract max
        vmax = float(v.max().item())
        denom = Decimal(0)
        num = None
        for vv, ii in zip(v.tolist(), i.tolist()):
            d = Decimal(str(vv - vmax)).exp()
            denom += d
            if int(ii) == int(token_id):
                num = d
        if num is None:
            # Shouldn't happen due to inclusion logic above, but keep safe.
            num = Decimal(0)
        return num / denom

def _decimal_sequence_probability(prompt_ids: list[int], generated_ids: list[int]) -> Decimal:
    # Score P(generated_ids | prompt_ids) under the same sampling distribution (temperature/top_k).
    with localcontext() as ctx_dec:
        ctx_dec.prec = 80
        ctx_dec.Emin = -999999999
        ctx_dec.Emax = 999999999
        seq_p = Decimal(1)
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
        for tok in generated_ids:
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            p_tok = _decimal_prob_for_token(logits[0].to(torch.float64).cpu(), int(tok), top_m=prob_decimal_top_m)
            seq_p *= p_tok
            idx = torch.cat([idx, torch.tensor([[int(tok)]], device=device, dtype=torch.long)], dim=1)
        return seq_p

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            if print_prob and not show_probs:
                # Use the model's generate() (which you modified) so it's easy to report probabilities.
                idx, prob = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, return_prob=True)
                p = float(prob[0].item())
                # Print with high precision; for very small temperatures this may be extremely close to 1.
                print(f"sequence_probability: {p:.30e}")
                print(f"one_minus_sequence_probability: {(1.0 - p):.30e}")
                if print_prob_decimal:
                    full_ids = idx[0].tolist()
                    prompt_len = x.size(1)
                    prompt_ids = full_ids[:prompt_len]
                    gen_ids = full_ids[prompt_len:]
                    dp = _decimal_sequence_probability(prompt_ids, gen_ids)
                    print(f"sequence_probability_decimal: {dp}")
                print(decode(idx[0].tolist()))
            else:
                # Inline generation loop so we can visualize probabilities each step.
                idx = x.clone()
                for step in range(max_new_tokens):
                    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / temperature
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)

                    if show_probs:
                        # Visualize the distribution we're sampling from (after temperature + top_k).
                        p = probs[0]
                        top_p, top_i = torch.topk(p, k=min(probs_top_n, p.numel()))
                        sampled_id = int(idx_next[0, 0].item())

                        save_path = None
                        if probs_save:
                            # Always write plots into out_dir, even for init_from=gpt2 (out_dir is still defined).
                            save_path = os.path.join(out_dir, "probs", f"sample{k}_step{step}.png")
                        _plot_top_probs(step=step, top_ids=top_i.tolist(), top_ps=top_p.tolist(), sampled_id=sampled_id, save_path=save_path)

                    idx = torch.cat((idx, idx_next), dim=1)

                print(decode(idx[0].tolist()))
            print('---------------')
