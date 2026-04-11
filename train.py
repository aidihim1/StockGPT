# train.py -- trains StockGPT on combined NSE returns data

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import StockGPT, StockGPTConfig

# ── Hyperparameters ────────────────────────────────────────────────────────────
BLOCK_SIZE  = 256
BATCH_SIZE  = 64
MAX_STEPS   = 10000
LR          = 3e-4
LR_MIN      = 3e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP   = 1.0
EVAL_EVERY  = 100
SAVE_EVERY  = 1000
DATA_FILE   = "dataset.csv"
CKPT_DIR    = "checkpoints"
# ──────────────────────────────────────────────────────────────────────────────

# 402-bin discretization: bins 1..400 are 50-bps intervals across [-100%, +100%]
BIN_EDGES = np.linspace(-1.0, 1.0, 401)   # 401 edges -> 400 interior bins

def tokenize_series(returns: np.ndarray) -> np.ndarray:
    """Convert float returns array to integer token array (drops NaN)."""
    valid = returns[~np.isnan(returns)]
    valid = np.clip(valid, -1.0 + 1e-9, 1.0 - 1e-9)
    tokens = np.digitize(valid, BIN_EDGES).astype(np.int16)  # values in [1, 400]
    return tokens


def load_sequences(path: str) -> list:
    """Load dataset.csv (long format) and return list of token arrays per stock."""
    print(f"Loading {path} ...")

    # Support both long format (dataset.csv) and wide format (combined_returns.csv)
    if "dataset.csv" in path or _is_long_format(path):
        df_long = pd.read_csv(path, parse_dates=["date"],
                              usecols=["date", "stock", "return_1d"])
        print(f"  Rows: {len(df_long):,}  |  Stocks: {df_long['stock'].nunique()}")
        sequences = []
        for stock, grp in df_long.groupby("stock"):
            returns = grp.sort_values("date")["return_1d"].values.astype(np.float64)
            tokens  = tokenize_series(returns)
            if len(tokens) >= BLOCK_SIZE + 1:
                sequences.append(tokens)
    else:
        # Legacy wide format fallback
        df = pd.read_csv(path, index_col=0)
        print(f"  Shape: {df.shape}")
        sequences = []
        for col in df.columns:
            tokens = tokenize_series(df[col].values)
            if len(tokens) >= BLOCK_SIZE + 1:
                sequences.append(tokens)

    print(f"  Stocks with enough data: {len(sequences)}")
    return sequences


def _is_long_format(path: str) -> bool:
    """Peek at header to detect long vs wide format."""
    with open(path, "r") as f:
        header = f.readline()
    return "stock" in header.lower() and "return" in header.lower()


def get_batch(sequences: list, batch_size: int, block_size: int, device: torch.device):
    """Sample a random batch from the sequence pool."""
    weights = np.array([len(s) - block_size for s in sequences], dtype=np.float64)
    weights /= weights.sum()

    stock_idxs = np.random.choice(len(sequences), size=batch_size, p=weights)

    x_list, y_list = [], []
    for si in stock_idxs:
        seq = sequences[si]
        max_start = len(seq) - block_size - 1
        start = np.random.randint(0, max_start + 1)
        chunk = seq[start: start + block_size + 1]
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])

    x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)
    return x, y


def get_lr(step: int) -> float:
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return LR_MIN + (LR - LR_MIN) * cosine


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Load data
    sequences = load_sequences(args.data)

    # Build model
    config = StockGPTConfig()
    model = StockGPT(config).to(device)
    model.get_num_params()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )

    # AMP scaler for FP16 (Quadro T2000 supports FP16)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"  Resumed at step {start_step}")

    log_rows = []
    print(f"\nTraining for {args.steps} steps, batch size {BATCH_SIZE} ...\n")

    for step in range(start_step, args.steps):
        # Update LR
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        x, y = get_batch(sequences, BATCH_SIZE, BLOCK_SIZE, device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item()

        if step % EVAL_EVERY == 0 or step == args.steps - 1:
            print(f"step {step:6d} | loss {loss_val:.4f} | lr {lr:.2e}")
            log_rows.append({"step": step, "loss": loss_val, "lr": lr})

        if (step + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"stockgpt_step_{step+1}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_val,
                "config": vars(config),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Save final model
    final_path = "stockgpt_final.pt"
    torch.save({
        "step": args.steps - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_val,
        "config": vars(config),
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    # Save training log
    pd.DataFrame(log_rows).to_csv("training_log.csv", index=False)
    print("Training log saved: training_log.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int,   default=MAX_STEPS)
    parser.add_argument("--data",    type=str,   default=DATA_FILE)
    parser.add_argument("--device",  type=str,   default="cuda")
    parser.add_argument("--resume",  type=str,   default=None)
    args = parser.parse_args()
    train(args)
