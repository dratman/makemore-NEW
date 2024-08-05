# MAKEMORE
# -----------------------
# You give this script some strings (one per line) and it will generate more strings like it.
# Uses super state of the art Transformer AI tech.
# This code is intended to be hackable. Tune it to your needs.
max_length = 210  # Set maximum input and output line length (was 120)
min_length = 25   # Set minimum input (training) sentence length.

import os
import sys
import time
from datetime import datetime
import math
import argparse
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import re

# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # These are default values, typically overridden by command-line args.
    block_size: int = None  # length of the input sequences of integers
    vocab_size: int = None  # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 12  # was 4
    n_embd: int = 144  # was 64
    n_embd2: int = 144  # was 64
    n_head: int = 8  # was 4
# end of class ModelConfig

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    # Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    # Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
# end of class NewGELU

class CausalSelfAttention(nn.Module): # Comments added by ChatGPT 2024-08-03 4:15AM

    # A vanilla multi-head masked self-attention layer with a projection at the end.
    # It is possible to use torch.nn.MultiheadAttention here but I am including an
    # explicit implementation to show that this is not too scary.

    def __init__(self, config):
        super().__init__()
        # Ensure that the embedding dimension is divisible by the number of heads.
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch.
        # The input embedding is split into three linear projections.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Causal mask to ensure that attention is only applied to the left in the input sequence.
        # `torch.tril` creates a lower triangular matrix of ones.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # Batch size (B), sequence length (T), embedding dimensionality (C or n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # Split the input into Q, K, V
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # Reshape and transpose K to (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # Reshape and transpose Q to (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # Reshape and transpose V to (B, nh, T, hs)

        # Causal self-attention:
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Scaled dot-product attention
        # Apply the causal mask to prevent attending to future tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # Softmax to get attention weights
        y = att @ v # Apply attention weights to values: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.c_proj(y)
        return y
# end of class CausalSelfAttention

class Block(nn.Module):
    # An unassuming Transformer block
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
# end of class Block

class Transformer(nn.Module):
    # Transformer Language Model, exactly as seen in GPT-2

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
# end of class Transformer

# helper functions for evaluating and sampling from the model
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    # Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    # the sequence max_new_tokens times, feeding the predictions back into the model each time.
    # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)

        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
# end of generate

# Sample from the model and pretty print the decoded samples.
def print_samples(num=10):

    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)

    top_k = args.top_k if args.top_k != -1 else None

    steps = train_dataset.get_output_length() - 1

    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')

    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        string_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(string_samp):
            train_samples.append(string_samp)
        elif test_dataset.contains(string_samp):
            test_samples.append(string_samp)
        else:
            new_samples.append(string_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for string in lst:
            print("---")
            print(string)
    print('-'*80)
# end of print_samples

# This function, evaluate, is designed to assess the performance of the model on a given dataset.
# "@torch.inference_mode()" is a decorator that puts PyTorch into inference mode.
# It disables gradient computation, which reduces memory usage and speeds up
# computation when you're not training.
# This function is typically used to assess how well the model is performing
# on a validation or test set during or after training. The mean loss gives
# an indication of the model's overall performance on the dataset.

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    # Put the model in evaluation mode, which can affect
    # certain layers like Dropout or BatchNorm.
    model.eval()

    # Create a DataLoader to efficiently iterate over the dataset in batches.
    # It shuffles the data and uses the specified batch size.
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)

    losses = []

    # Loop over batches from the DataLoader; move each tensor in the batch to the
    # specified device (CPU/GPU), and unpack the input (X) and target (Y) from the batch.
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # Run the model on the input X and compute the loss against the target Y.
        logits, loss = model(X, Y)

        # Add the loss value for this batch to the list of losses.
        losses.append(loss.item())

        # Break the loop if we've processed the maximum number of batches specified.
        if max_batches is not None and i >= max_batches:
            break
    # Compute the mean loss across all batches.
    mean_loss = torch.tensor(losses).mean().item()

    model.train() # reset model back to training mode
    return mean_loss
# end of evaluate

# ------------------------------
# helper functions for creating the training and test Datasets that emit strings
class CharDataset(Dataset):

    def __init__(self, strings, chars, max_string_length):
        self.strings = strings
        self.chars = chars
        self.max_string_length = max_string_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.strings)

    def contains(self, string):
        return string in self.strings

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_string_length + 1 # <START> token followed by strings

    def encode(self, string):
        ix = torch.tensor([self.stoi[w] for w in string], dtype=torch.long)
        return ix

    def decode(self, ix):
        string = ''.join(self.itos[i] for i in ix)
        return string

    def __getitem__(self, idx):
        string = self.strings[idx]
        ix = self.encode(string)
        x = torch.zeros(self.max_string_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_string_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y
# end of class CharDataset

# def extract_sentences(text, max_length):
#     # Regex pattern to find sentences
#     sentence_pattern = re.compile(r'([A-Z][^.!?]{0,' + str(max_length - 2) + r'}[.!?])')
#     sentences = sentence_pattern.findall(text)
#     return [s.strip() for s in sentences if len(s) <= max_length]
# def extract_sentences(text, max_length):
#     # Regex pattern to find sentences
#     sentence_pattern = re.compile(r'([A-Z][^.!?]{0,' + str(max_length - 2) + r'}[.!?])')
#     sentences = sentence_pattern.findall(text)
#     return [s.strip() for s in sentences if len(s) <= max_length]
#     import re

def extract_sentences(text, max_length):
    # Regex pattern to find sentences
    sentence_pattern = re.compile(r'([A-Z][^.!?]{0,' + str(max_length - 2) + r'}[.!?])')
    ascii_pattern = re.compile(r'^[\x20-\x7E]+$')  # Pattern to match only printable ASCII characters
    sentences = sentence_pattern.findall(text)
    filtered_sentences = []
    for s in sentences:
        if len(s) <= max_length and len(s)>= min_length and ascii_pattern.match(s):
            filtered_sentences.append(s.strip())
            # Also print to stderr
            print(s.strip(), file=sys.stderr) # DEBUG ONLY
    return filtered_sentences

def create_datasets(input_file):
    # Preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()

    # Extract sentences based on the new criteria
    strings = extract_sentences(data, 109)

    # Get rid of any empty strings
    strings = [w for w in strings if w]

    chars = sorted(list(set(''.join(strings))))  # All the possible characters
    max_string_length = max(len(w) for w in strings)
    print(f"number of examples in the dataset: {len(strings)}")
    print(f"max string length: {max_string_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    # Partition the input data into a training and the test set
    test_set_size = min(1000, int(len(strings) * 0.1))  # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(strings)).tolist()
    train_strings = [strings[i] for i in rp[:-test_set_size]]
    test_strings = [strings[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_strings)} training examples and {len(test_strings)} test examples")

    # Wrap in dataset objects
    train_dataset = CharDataset(train_strings, chars, max_string_length)
    test_dataset = CharDataset(test_strings, chars, max_string_length)

    return train_dataset, test_dataset


class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
# end of class InfiniteDataLoader

# ------------------------------
if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with strings one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=args.n_layer, n_head=args.n_head,
                         n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only:  # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    sample_counter = 0

    step = 0
    while True:
        t0 = time.time()
        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        sample_counter += X.size(0)  # Increment by the actual batch size
        # feed into the model
        logits, loss = model(X, Y)
        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()

        optimizer.step()
        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()
        # logging

        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")
        # evaluate the model

        if step > 0 and step % 500 == 0:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime(">>>>>   %Y-%m-%d %H:%M")
            print(f"{formatted_datetime} | Step: {step} | Samples processed: {sample_counter}")

            # prepare to print loss report
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)

            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.add_scalar("Learning rate", args.learning_rate, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                out_path = os.path.join(args.work_dir, "model.pt")
                torch.save(model.state_dict(), out_path)

        # sample from the model

        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1

        # termination conditions

        if args.max_steps >= 0 and step >= args.max_steps:
            break

        # Inside the training loop, after a certain number of steps (e.g., every 100 steps)
        if step % 100 == 0:
            if os.path.exists("stop.txt"):
                print("stop.txt file now exists. Stopping the training.")
                break

# end while True
