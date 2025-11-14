import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Holofractal Transformer Layer
# -------------------------

class HolofractalTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )

        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # Learnable global token (boundary representation)
        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.global_token)

    def forward(self, x, attn_mask=None):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Append global token
        g_tok = self.global_token.expand(batch_size, 1, -1)   # (B, 1, D)
        x_with_global = torch.cat([g_tok, x], dim=1)          # (B, 1+L, D)

        # 2. Self-attention over [global + tokens]
        attn_out, _ = self.self_attn(
            x_with_global, x_with_global, x_with_global,
            attn_mask=attn_mask, need_weights=False
        )

        # Residual around attention
        x_with_global_orig = torch.cat([g_tok, x], dim=1)
        attn_res = x_with_global_orig + self.dropout_attn(attn_out)

        # Hyperspherical normalization (boundary projection)
        attn_res_norm = F.normalize(attn_res, p=2, dim=-1)

        # 3. Feed-forward (on global + tokens)
        ff_in = self.norm1(attn_res_norm)             # (B, 1+L, D)
        ff_mid = F.gelu(self.fc1(ff_in))
        ff_mid = self.dropout_ffn(ff_mid)
        ff_out = self.fc2(ff_mid)

        # Residual around FFN
        ff_res = attn_res_norm + self.dropout_ffn(ff_out)

        # Normalize back onto hypersphere
        ff_res_norm = F.normalize(ff_res, p=2, dim=-1)

        # Split back out
        g_out = ff_res_norm[:, :1, :]                  # (B, 1, D)
        x_out = ff_res_norm[:, 1:, :]                  # (B, L, D)

        return x_out, g_out


# -------------------------
# Holofractal Transformer Model
# -------------------------

class HolofractalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, dim_ff=128,
                 num_layers=2, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList([
            HolofractalTransformerLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm_final = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        """
        input_ids: (B, L)
        """
        bsz, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len

        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]

        g = None
        for layer in self.layers:
            x, g = layer(x)

        x = self.norm_final(x)
        logits = self.out_proj(x)   # (B, L, vocab)
        return logits, g            # g is (B, 1, D)


# -------------------------
# Synthetic Training Loop (Proof of Concept)
# -------------------------

def generate_batch(batch_size, seq_len, vocab_size, device):
    """
    Generate a batch of sequences and next-token targets.
    Task: predict the sequence shifted by 1 position.
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Next-token prediction: y[t] = x[t+1], last position wraps to first
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


def train_poc(
    vocab_size=50,
    d_model=64,
    n_heads=4,
    dim_ff=128,
    num_layers=2,
    max_seq_len=16,
    batch_size=32,
    num_steps=500,
    lr=3e-4,
    print_every=50,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HolofractalTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        dim_ff=dim_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, num_steps + 1):
        input_ids, targets = generate_batch(
            batch_size, max_seq_len, vocab_size, device
        )
        logits, g = model(input_ids)  # logits: (B, L, V)

        loss = criterion(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % print_every == 0 or step == 1:
            print(f"Step {step}/{num_steps}, loss = {loss.item():.4f}")

    return model


def demo_after_training(model, vocab_size=50, seq_len=10, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        x = torch.randint(0, vocab_size, (1, seq_len), device=device)
        logits, g = model(x)
        preds = torch.argmax(logits, dim=-1)

    print("\nDemo sequence:")
    print("Input:     ", x[0].cpu().tolist())
    print("Target:    ", torch.roll(x, shifts=-1, dims=1)[0].cpu().tolist())
    print("Predicted: ", preds[0].cpu().tolist())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = train_poc(
        vocab_size=50,
        d_model=64,
        n_heads=4,
        dim_ff=128,
        num_layers=2,
        max_seq_len=16,
        batch_size=32,
        num_steps=400,
        lr=3e-4,
        print_every=50,
        device=device,
    )

    demo_after_training(model, vocab_size=50, seq_len=10, device=device)
