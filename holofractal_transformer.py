import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class HoloFractalTransformer(nn.Module):
    """Transformer model with a global boundary token for multi-scale financial data."""
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, n_assets=1, max_time_steps=256):
        super().__init__()
        # Embedding layers for value features, asset IDs, and positional encoding
        self.value_embed = nn.Linear(5, d_model)                    # projects OHLCV (5 features) to d_model
        self.asset_embed = nn.Embedding(n_assets, d_model)          # asset ID embedding
        self.pos_embed   = nn.Embedding(max_time_steps, d_model)    # positional encoding (learnable embeddings)
        # Boundary token embedding (learned parameter representing global context)
        self.boundary_embed = nn.Parameter(torch.randn(d_model))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Prediction heads
        self.return_head = nn.Linear(d_model, 3)   # predicts 1-day, 5-day, 20-day returns
        self.regime_head = nn.Linear(d_model, 2)   # predicts volatility regime (2 classes)

    def forward(self, ohlcv_sequence, asset_id):
        """
        Forward pass for a batch of sequences.
        - ohlcv_sequence: Tensor of shape (batch, seq_len, 5) with OHLCV data.
        - asset_id: Tensor of shape (batch,) containing asset IDs for each sequence in the batch.
        Returns: (returns_pred, regime_logit, boundary_embedding) for each sequence.
        """
        batch_size, seq_len, _ = ohlcv_sequence.shape
        # Compute embeddings for OHLCV values, asset IDs, and positions
        value_emb = self.value_embed(ohlcv_sequence)                # (batch, seq_len, d_model)
        asset_emb = self.asset_embed(asset_id).unsqueeze(1)         # (batch, 1, d_model) for each asset, to broadcast
        asset_emb = asset_emb.expand(batch_size, seq_len, asset_emb.size(-1))  # (batch, seq_len, d_model)
        positions = torch.arange(seq_len, device=ohlcv_sequence.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions).expand(batch_size, seq_len, -1)    # (batch, seq_len, d_model)
        # Combine embeddings (element-wise sum)
        token_emb = value_emb + asset_emb + pos_emb
        # Replace the last token's embedding with the special boundary token embedding (plus asset bias, no positional bias)
        token_emb[:, -1, :] = self.boundary_embed + self.asset_embed(asset_id)
        # Pass through transformer encoder
        encoded = self.transformer(token_emb)       # (batch, seq_len, d_model)
        # Extract the boundary token's output (we placed it at end of sequence)
        boundary_out = encoded[:, -1, :]            # (batch, d_model)
        # Constrain boundary output to unit hypersphere (normalize)
        boundary_norm = boundary_out / (boundary_out.norm(dim=1, keepdim=True) + 1e-8)
        # Prediction heads use the normalized boundary embedding
        returns_pred = self.return_head(boundary_norm)   # (batch, 3)
        regime_logit = self.regime_head(boundary_norm)   # (batch, 2)
        return returns_pred, regime_logit, boundary_norm

def generate_synthetic_data(n_assets=3, n_days=120, sigma_low=0.01, sigma_high=0.04, switch_prob=0.1):
    """
    Generate synthetic OHLCV data for multiple assets with high-vol and low-vol regimes.
    Each asset starts at a random price ~100 and undergoes a random walk.
    Volatility (std of returns) switches between sigma_low and sigma_high with probability `switch_prob` each day.
    Returns: tuple (data_by_asset, regimes_by_asset), where:
      - data_by_asset is a list of np.ndarrays of shape (n_days, 5) [OHLCV for each day].
      - regimes_by_asset is a list of np.ndarrays of shape (n_days,) [regime label 0/1 for each day].
    """
    data_by_asset = []
    regimes_by_asset = []
    for asset_idx in range(n_assets):
        prices = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        regimes = []
        # Initialize price and regime
        price = 100.0 + np.random.rand() * 20.0       # initial price between 100 and 120
        regime = np.random.choice([0, 1])             # 0 = low-vol, 1 = high-vol
        for day in range(n_days):
            # Possibly switch regime
            if day > 0 and np.random.rand() < switch_prob:
                regime = 1 - regime
            sigma = sigma_high if regime == 1 else sigma_low
            # Simulate daily return from normal distribution
            daily_ret = np.random.normal(0, sigma)
            open_price = price
            close_price = price * (1 + daily_ret)
            # Simulate high/low as deviations around open/close
            noise_up = np.random.rand() * sigma * 2.0    # fraction of open price
            noise_down = np.random.rand() * sigma * 2.0
            high_price = max(open_price, close_price) * (1 + noise_up)
            low_price  = min(open_price, close_price) * (1 - noise_down)
            if low_price < 0:
                low_price = 0.0  # avoid negative (for extreme cases)
            # Simulate volume (higher on high-vol days)
            base_vol = 1000.0 if regime == 0 else 3000.0
            volume = base_vol * (0.8 + 0.4 * np.random.rand())
            # Record the day's data
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            regimes.append(regime)
            # Update price for next day
            price = close_price
        # Convert to numpy arrays
        data = np.column_stack([opens, highs, lows, closes, volumes]).astype(np.float32)
        data_by_asset.append(data)
        regimes_by_asset.append(np.array(regimes, dtype=np.int64))
    return data_by_asset, regimes_by_asset

# Training configuration
N_ASSETS = 3
DAYS_PER_ASSET = 120
WINDOW_SIZE = 20               # length of input sequence (in trading days)
HORIZON_1 = 1
HORIZON_5 = 5
HORIZON_20 = 20
NUM_EPOCHS = 5
BATCH_SIZE = 1                 # using batch_size=1 for simplicity in computing fractal losses
LEARNING_RATE = 1e-3

if __name__ == "__main__":
    # 1. Generate synthetic dataset
    data_by_asset, regimes_by_asset = generate_synthetic_data(n_assets=N_ASSETS, n_days=DAYS_PER_ASSET)
    samples = []  # will hold tuples of (input_sequence, asset_id, target_returns, target_regime)
    for asset_id, (ohlcv, regime_seq) in enumerate(zip(data_by_asset, regimes_by_asset)):
        n_days = ohlcv.shape[0]
        # Create sliding windows of length WINDOW_SIZE
        for start in range(0, n_days - WINDOW_SIZE - max(HORIZON_5, HORIZON_20)):
            end = start + WINDOW_SIZE
            seq_data = ohlcv[start:end]            # shape (20, 5)
            # Compute future returns for horizons
            # (We assume availability of future prices up to start+39 for 20-day return)
            price_t = ohlcv[end-1, 3]              # closing price at end of input window (day index end-1)
            price_t1 = ohlcv[end + HORIZON_1 - 1, 3]   # close price 1 day after window
            price_t5 = ohlcv[end + HORIZON_5 - 1, 3]   # close price 5 days after window
            price_t20 = ohlcv[end + HORIZON_20 - 1, 3] # close price 20 days after window
            ret1  = (price_t1  - price_t) / (price_t + 1e-8)
            ret5  = (price_t5  - price_t) / (price_t + 1e-8)
            ret20 = (price_t20 - price_t) / (price_t + 1e-8)
            target_returns = np.array([ret1, ret5, ret20], dtype=np.float32)
            # Regime label for current period (we use regime of the last day in the input window)
            target_regime = regime_seq[end-1]
            samples.append((seq_data, asset_id, target_returns, target_regime))
    # Convert list of samples to tensors for easier indexing
    # (For large data, we would create a DataLoader to load in batches, but data is small here)
    X = torch.tensor([s[0] for s in samples])              # shape (N_samples, 20, 5)
    X_asset = torch.tensor([s[1] for s in samples])        # shape (N_samples,)
    y_returns = torch.tensor([s[2] for s in samples])      # shape (N_samples, 3)
    y_regime  = torch.tensor([s[3] for s in samples])      # shape (N_samples,) (0 or 1)
    # 2. Initialize model and optimizer
    model = HoloFractalTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                                   n_assets=N_ASSETS, max_time_steps=WINDOW_SIZE+1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Loss functions
    regression_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()
    # 3. Training loop
    model.train()
    for epoch in range(1, NUM_EPOCHS+1):
        # Shuffle training samples each epoch
        indices = torch.randperm(X.size(0))
        total_loss = 0.0
        for idx in indices:
            seq = X[idx].unsqueeze(0)                 # shape (1, 20, 5)
            asset = X_asset[idx].unsqueeze(0)         # shape (1,)
            target_ret = y_returns[idx].unsqueeze(0)  # shape (1, 3)
            target_reg = y_regime[idx].unsqueeze(0)   # shape (1,)
            # Forward pass on full sequence
            pred_ret, pred_reg, full_emb = model(seq, asset)
            # Calculate primary losses
            loss_ret = regression_loss_fn(pred_ret, target_ret)
            loss_reg = classification_loss_fn(pred_reg, target_reg)
            # Fractal consistency loss: align embeddings across scales
            # Compute 5-day segment embeddings and 1-day embeddings
            # Split the 20-day sequence into four 5-day segments
            fractal_loss = 0.0
            seq_np = seq.detach().cpu().numpy().reshape(WINDOW_SIZE, 5)
            # For 5-day segments within the 20-day window
            segment_embeds = []
            for i in range(0, WINDOW_SIZE, 5):
                seg = seq[:, i:i+5, :]                 # (1, 5, 5)
                _, _, seg_emb = model(seg, asset)      # boundary embedding for this 5-day segment
                segment_embeds.append(seg_emb)         # list of shape (1, d_model)
                # For each single day in this segment, get boundary embedding (the day itself)
                day_embeds = []
                for j in range(i, i+5):
                    day_seq = seq[:, j:j+1, :]         # (1, 1, 5)
                    _, _, day_emb = model(day_seq, asset)
                    day_embeds.append(day_emb)
                # Compute average of the 5 daily embeddings
                days_avg = torch.stack(day_embeds).mean(dim=0)  # (1, d_model)
                # Alignment loss between segment embedding and average of day embeddings
                fractal_loss += regression_loss_fn(seg_emb, days_avg)
            # Compute average of the four 5-day segment embeddings
            segments_avg = torch.stack(segment_embeds).mean(dim=0)  # (1, d_model)
            # Alignment loss between full 20-day embedding and average of 5-day segment embeddings
            fractal_loss += regression_loss_fn(full_emb, segments_avg)
            # Total loss
            loss = loss_ret + loss_reg + fractal_loss
            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(indices)
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Training loss: {avg_loss:.4f}")
