#!/usr/bin/env python3
"""
Clyde TinyGPT Model — Core Architecture

A minimal, efficient GPT-style transformer language model designed for:
- Local CPU inference (no GPU required)
- Clyde AI system foundation
- Tool calling and function integration
- Multi-language support
- Read/write/speak capabilities

ARCHITECTURE: 2-layer transformer with causal self-attention
PARAMETERS: ~5K (small enough for CPU training on Raspberry Pi)
CONTEXT: 32 tokens
VOCABULARY: 65 characters (alphanumeric + punctuation)
"""

import math
import numpy as np

# ── Hyperparameters ──────────────────────────────────────────────────────

DROP_OUT = 0.1          # Dropout probability
LR = 3e-4               # Learning rate for AdamW
BATCH_SIZE = 8          # Training batch size
MAX_STEPS = 5000        # Default training steps
GRAD_CLIP = 1.0         # Gradient clipping norm

# Model dimensions (TinyGPT scale)
N_EMBED = 64            # Token embedding dimension
N_HEAD = 4              # Number of attention heads
N_LAYER = 2             # Number of transformer layers
BLOCK_SIZE = 32         # Context window size (tokens)
VOCAB_SIZE = 65         # Character vocabulary size

# ── Device Configuration ─────────────────────────────────────────────────

def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mpis") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
print(f"🤖 TinyGPT using device: {DEVICE}")

# ── Vocabulary ───────────────────────────────────────────────────────────

# 65 characters: space + 26 uppercase + 26 lowercase + digits + some punctuation
CHAR_SET = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
CHAR_SET += "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"

char_to_idx = {ch: i for i, ch in enumerate(CHAR_SET)}
idx_to_char = {i: ch for i, ch in enumerate(CHAR_SET)}
VOCAB_SIZE = len(CHAR_SET)

def encode(text):
    """Convert string to list of integers (character indices)."""
    return [char_to_idx.get(ch, char_to_idx[' ']) for ch in text]

def decode(indices):
    """Convert list of integers back to string."""
    return ''.join(idx_to_char.get(i, ' ') for i in indices)

# ── Positional Encoding ──────────────────────────────────────────────────

def create_positional_encodings(block_size, n_embd):
    """Create sinusoidal positional encodings (numpy arrays)."""
    pe = np.zeros((block_size, n_embd))
    position = np.arange(0, block_size, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, n_embd, 2, dtype=np.float32) * 
                     (-math.log(10000.0) / n_embd))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe  # Shape: (block_size, n_embd)

# ── Causal Mask ──────────────────────────────────────────────────────────

def create_causal_mask(seq_len):
    """Create a causal (upper-triangular) mask for self-attention."""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return torch.from_numpy(mask == 0).bool()  # True = keep, False = mask

# ── Model Components ─────────────────────────────────────────────────────

class CausalSelfAttention(torch.nn.Module):
    """Causal self-attention block for TinyGPT."""
    
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        
        # Key, Query, Value projections
        self.key = torch.nn.Linear(n_embd, n_embd)
        self.query = torch.nn.Linear(n_embd, n_embd)
        self.value = torch.nn.Linear(n_embd, n_embd)
        
        # Output projection
        self.out_proj = torch.nn.Linear(n_embd, n_embd)
        
        # Causal mask register
        self.register_buffer("causal_mask", 
                           torch.tril(torch.ones(block_size, block_size))
                          .expand(n_embd, block_size, block_size)
                          .reshape(-1, block_size, block_size))
    
    def forward(self, x):
        B, T, C = x.size()  # Batch, Time, Channel (embed dim)
        
        # Compute key, query, value
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Attention scores (matmul query and transposed key)
        # q: (B, n_head, T, head_size), k.transpose(-2,-1): (B, n_head, head_size, T)
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)  # (B, n_head, T, T)
        
        # Apply causal mask
        mask = self.causal_mask[:, :, :T, :T]  # (1, n_head, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        att = torch.nn.functional.softmax(att, dim=-1)
        att = torch.nn.functional.dropout(att, p=DROP_OUT, training=self.training)
        
        # Apply to values
        y = att @ v  # (B, n_head, T, head_size)
        
        # Reassemble and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        
        return y

class MLPPBlock(torch.nn.Module):
    """MLP (Multi-Layer Perceptron) block with GELU activation."""
    
    def __init__(self, n_embd, expansion=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, n_embd * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(n_embd * expansion, n_embd),
            torch.nn.Dropout(DROP_OUT)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(torch.nn.Module):
    """Complete transformer block: attention + MLP with residual connections."""
    
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = torch.nn.LayerNorm(n_embd)
        self.mlp = MLPPBlock(n_embd)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln_1(x))
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x

# ── Full Model ───────────────────────────────────────────────────────────

class TinyGPT(torch.nn.Module):
    """Minimal GPT-style language model."""
    
    def __init__(self, vocab_size=VOCAB_SIZE, n_embed=N_EMBED, n_head=N_HEAD,
                 n_layer=N_LAYER, block_size=BLOCK_SIZE, dropout=DROP_OUT):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        
        # Token embedding + positional encoding
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embed)
        self.position_embedding = torch.nn.Embedding(block_size, n_embed)
        
        # Dropout after embedding
        self.drop = torch.nn.Dropout(dropout)
        
        # Transformer block stack
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(n_embed, n_head, block_size) 
            for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = torch.nn.LayerNorm(n_embed)
        
        # Language modeling head (weight tying with token embedding)
        self.lm_head = torch.nn.Linear(n_embed, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
    
    def forward(self, idx, targets=None):
        """
        idx: (B, T) array of integers in [0, vocab_size)
        targets: (B, T) optional, for computing loss
        """
        B, T = idx.shape
        
        # Precompute positional embeddings (or use learned)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        
        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(pos)  # (1, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Shift logits and targets for next-token prediction
            shift_logits = logits[:, :-1, :]  # (B, T-1, vocab_size)
            shift_targets = targets[:, 1:]    # (B, T-1)
            
            # Flatten for cross-entropy
            loss_fct = torch.nn.CrossEntropyLoss()
            B_T_v = shift_logits.view(-1, vocab_size)
            B_T_t = shift_targets.view(-1)
            
            # Ignore index = 0 (padding/unknown)
            loss = loss_fct(B_T_v, B_T_t)  # scalar
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.7, top_k=None):
        """
        Generate new tokens given starting sequence.
        
        idx: (B, T) starting sequence
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature (lower = more deterministic)
        top_k: if set, only consider top k tokens ( nucleus sampling)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, VOCAB_SIZE))
                logits = torch.where(logits < v[:, [-1], :], float('-inf'), logits)
            
            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        
        return idx

# ── Training Function ────────────────────────────────────────────────────

def train_tiny_gpt(model, train_texts, lr=LR, steps=MAX_STEPS, 
                   batch_size=BATCH_SIZE, device=DEVICE, save_path=None):
    """
    Train the TinyGPT model on provided text data.
    
    Args:
        model: TinyGPT instance
        train_texts: List of training strings
        lr: Learning rate
        steps: Number of training steps
        batch_size: Batch size
        device: Training device
        save_path: Path to save checkpoint
    
    Returns:
        model: Trained model
        history: Dict with loss progression
    """
    model.train()
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Prepare all training data
    all_ids = []
    for text in train_texts:
        ids = encode(text)
        all_ids.extend(ids)
    
    # Convert to tensor
    data = torch.tensor(all_ids, dtype=torch.long, device=device)
    
    # Training history
    history = {"loss": []}
    
    # ── Training Loop ──────────────────────────────────────────────────
    for step in range(steps):
        # Create mini-batch
        # Randomly sample a chunk of sequence
        max_start = data.size(0) - block_size
        if max_start < 1:
            break
        start_idx = torch.randint(0, max_start, (batch_size,))
        
        batch = []
        for i in start_idx:
            chunk = data[i:i+block_size+1]  # input + target
            x = chunk[:-1]  # inputs
            y = chunk[1:]   # targets
            batch.append((x, y))
        
        if not batch:
            continue
        
        # Pad batch to same length (all same size due to block_size)
        x_batch = torch.stack([b[0] for b in batch])
        y_batch = torch.stack([b[1] for b in batch])
        
        # Forward + backward
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        # Record loss
        if step % 100 == 0 or step == steps - 1:
            history["loss"].append(loss.item())
            print(f"  Step {step:5d} | Loss: {loss.item():.4f} | "
                  f"PPL: {math.exp(min(loss.item(), 100)):.2f}")
    
    # ── Save Checkpoint ────────────────────────────────────────────────
    if save_path:
        # Save model state, config, and vocab
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": VOCAB_SIZE,
                "n_embed": N_EMBED,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
                "block_size": BLOCK_SIZE,
                "dropout": DROP_OUT
            },
            "vocab_map": {
                "char_to_idx": {k: v for k, v in char_to_idx.items()},
                "idx_to_char": {v: k for k, v in idx_to_char.items()}
            },
            "steps": steps,
            "final_loss": loss.item(),
            "device": device
        }
        torch.save(checkpoint, save_path)
        print(f"  ✅ Model saved to: {save_path}")
    
    return model, history

# ── Inference / Generation ────────────────────────────────────────────────

def sample_text(model, prime_text="Hello,", max_new_tokens=50, 
                temperature=0.7, top_k=None, device=DEVICE):
    """
    Generate text starting from prime_text.
    
    Args:
        model: Trained TinyGPT model
        prime_text: Starting text/prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        device: Computation device
    
    Returns:
        Generated text string
    """
    model.eval()
    model.to(device)
    
    # Encode the priming text
    encoded = encode(prime_text)
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    
    # Generate
    generated = model.generate(idx, max_new_tokens=max_new_tokens,
                              temperature=temperature, top_k=top_k, device=device)
    
    # Decode all tokens
    all_ids = generated.squeeze(0).cpu().tolist()
    # Decode only the NEW tokens (after the prime)
    new_ids = all_ids[len(encoded):]
    generated_text = decode(new_ids)
    
    return prime_text + generated_text

# ── Utility Functions ────────────────────────────────────────────────────

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model):
    """Print model architecture summary."""
    n_params = count_parameters(model)
    print(f"📊 TinyGPT Model Summary:")
    print(f"   • Parameters: {n_params:,}")
    print(f"   • Vocabulary size: {VOCAB_SIZE}")
    print(f"   • Embedding dim: {N_EMBED}")
    print(f"   • Layers: {N_LAYER}")
    print(f"   • Heads: {N_HEAD}")
    print(f"   • Context window: {BLOCK_SIZE} tokens")
    print(f"   • Device: {DEVICE}")
    print(f"   • Dropout: {DROP_OUT}")

# ── Main Execution (Demo) ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Clyde TinyGPT Model — Initializing")
    print("=" * 60)
    
    # Create model instance
    model = TinyGPT()
    
    # Print model info
    print_model_info(model)
    
    # Small demo training
    demo_texts = [
        "The student wrote a clear sentence.",
        "A paragraph should have one idea.",
        "def add(a, b): return a + b",
        "Hello, Clyde. How are you today?",
        "Python is a great programming language.",
        "The cat sat on the mat.",
    ]
    
    print(f"\n📚 Training on {len(demo_texts)} demo texts...")
    model, history = train_tiny_gpt(
        model, 
        demo_texts, 
        steps=500,  # Quick demo - use 5000+ for real training
        save_path=" Clyde_model_demo.pt"
    )
    
    # Generate some text
    print(f"\n🎯 Generating text...")
    result = sample_text(model, prime_text="The student", 
                        max_new_tokens=30, temperature=0.7)
    print(f"  Prime: 'The student'")
    print(f"  Generated: {result}")
    
    print(f"\n✅ Demo complete!")