# gpt_model.py
# Educational-only code for learning purposes, not intended for commercial use.
# This script implements a simplified GPT model using PyTorch.

import torch
import torch.nn as nn
from torch.nn import functional as F

# Define model parameters for customization
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Set a random seed for reproducibility
torch.manual_seed(1337)

# Load and prepare data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character mappings for encoding/decoding
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train/validation data split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Function to load batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Transformer components
class Head(nn.Module):
    """One head of self-attention"""
    # Initializations and forward method definitions go here as in original script
    ...

# Define the main GPT model
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Set up token embeddings, positional embeddings, and transformer blocks
        ...

    def forward(self, idx, targets=None):
        # Define the forward pass to handle inputs and outputs
        ...

    def generate(self, idx, max_new_tokens):
        # Define the text generation logic
        ...

# Instantiate model and optimizer
model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch and perform gradient descent
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate sample text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
