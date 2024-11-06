# bigram_model.py
# Educational-only code for learning purposes, not intended for commercial use.
# This is a simple Bigram Language Model built with PyTorch.

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters: customize based on your computational resources
batch_size = 32      # Number of sequences processed in parallel
block_size = 8       # Maximum context length for predictions
max_iters = 3000     # Total iterations for training
eval_interval = 300  # Interval to evaluate the loss
learning_rate = 1e-2 # Learning rate for optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check for GPU availability
eval_iters = 200     # Number of evaluations during validation

# Set a random seed for reproducibility
torch.manual_seed(1337)

# Load and prepare data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get unique characters in the text and create mappings to/from integers
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # char-to-integer map
itos = {i: ch for i, ch in enumerate(chars)}  # integer-to-char map
encode = lambda s: [stoi[c] for c in s]       # String to integer list
decode = lambda l: ''.join([itos[i] for i in l])  # Integer list to string

# Train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # Use 90% of data for training
train_data = data[:n]
val_data = data[n:]

# Function to load batches of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# Estimate loss for evaluation purposes
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

# Define a Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # Logits shape: (B, T, C)

        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)  # Apply softmax for probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # Append sampled index
        return idx

# Instantiate the model and optimizer
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    # Evaluate loss on train/val sets periodically
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Compute loss and update model parameters
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
