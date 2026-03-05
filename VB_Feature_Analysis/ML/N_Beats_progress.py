#!/usr/bin/env python3
"""
Optimized N-BEATS pipeline with full terminal progress display.
Now updated with latest PyTorch AMP API (no deprecation warnings).

Features:
 - Streaming dataset (cached parquet loading)
 - TQDM progress bars everywhere
 - AMP accelerated training (torch.amp.autocast)
 - No deprecated API usage
 - Prints GPU memory usage each epoch
 - Outputs models/nbeats.pt and signals/dayX_signals.csv
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = Path("/data/quant14/EBX")
MODEL_DIR = Path("./models")
SIGNAL_DIR = Path("./signals")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

PRICE_COL = "Price"
TIME_COL = "Time"

INPUT_WINDOW = 600
FORECAST_HORIZON = 60

BATCH_SIZE = 1024
EPOCHS = 5
LR = 1e-4
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DAYS = range(0, 358)
VAL_DAYS   = range(358, 409)
TEST_DAYS  = range(409, 510)

LONG_THRESHOLD = 5e-4
SHORT_THRESHOLD = -5e-4

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# FAST CACHED DAY LOADER
# -------------------------
day_cache = {}

def load_day_prices(day_index):
    if day_index in day_cache:
        return day_cache[day_index]

    fp = DATA_DIR / f"day{day_index}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

    df = pd.read_parquet(fp, columns=[TIME_COL, PRICE_COL])
    price_arr = df[PRICE_COL].to_numpy(dtype=float)

    day_cache[day_index] = price_arr  # cache for speed
    return price_arr

# -------------------------
# STREAMING DATASET
# -------------------------
class StreamingDataset(Dataset):
    def __init__(self, day_list):
        self.day_list = list(day_list)
        self.index = []

        print("Indexing samples...")
        for d in tqdm(self.day_list, desc="Indexing Days", ncols=100):
            prices = load_day_prices(d)
            n = len(prices)

            for i in range(INPUT_WINDOW, n - FORECAST_HORIZON + 1):
                self.index.append((d, i))

        print(f"Total samples indexed: {len(self.index):,}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        d, i = self.index[idx]
        prices = load_day_prices(d)

        X = prices[i-INPUT_WINDOW:i].astype(np.float32)
        Y = prices[i:i+FORECAST_HORIZON].astype(np.float32)

        return torch.from_numpy(X), torch.from_numpy(Y)

# -------------------------
# LIGHT N-BEATS MODEL
# -------------------------
class NBeatsBlock(nn.Module):
    def __init__(self, input_size=INPUT_WINDOW, theta_size=128, width=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, theta_size),
            nn.ReLU()
        )
        self.backcast = nn.Linear(theta_size, input_size)
        self.forecast = nn.Linear(theta_size, FORECAST_HORIZON)

    def forward(self, x):
        theta = self.fc(x)
        b = self.backcast(theta)
        f = self.forecast(theta)
        return b, f


class NBeats(nn.Module):
    def __init__(self, blocks=6, width=512, theta_size=128):
        super().__init__()
        self.blocks = nn.ModuleList(
            [NBeatsBlock(theta_size=theta_size, width=width) for _ in range(blocks)]
        )

    def forward(self, x):
        residual = x
        forecast_sum = 0
        for blk in self.blocks:
            backcast, forecast = blk(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum

# -------------------------
# TRAINING
# -------------------------
def train_model():
    print("Preparing datasets...")
    train_ds = StreamingDataset(TRAIN_DAYS)
    val_ds = StreamingDataset(VAL_DAYS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = NBeats().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, EPOCHS + 1):
        # ------------------------- TRAIN
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train", ncols=120)

        for X, Y in pbar:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            opt.zero_grad()

            with torch.amp.autocast("cuda"):
                pred = model(X)
                loss = loss_fn(pred, Y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_train = train_loss / len(train_loader)

        # ------------------------- VAL
        model.eval()
        val_loss = 0.0
        vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} Val", ncols=120)

        with torch.no_grad():
            for X, Y in vbar:
                X = X.to(DEVICE)
                Y = Y.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    pred = model(X)
                    loss = loss_fn(pred, Y)
                val_loss += loss.item()
                vbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_val = val_loss / len(val_loader)

        # GPU STATS
        if DEVICE == "cuda":
            mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_res = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[Epoch {epoch}] Train={avg_train:.6f}  Val={avg_val:.6f} | GPU Used={mem_used:.2f}GB  Reserved={mem_res:.2f}GB")
        else:
            print(f"[Epoch {epoch}] Train={avg_train:.6f}  Val={avg_val:.6f}")

    model_path = MODEL_DIR / "nbeats.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved → {model_path}")

# -------------------------
# SLOPE UTILS
# -------------------------
def linreg_slope(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return float(model.coef_[0])

def slope_to_return(slope, price):
    if price == 0:
        return np.nan
    return (slope * FORECAST_HORIZON) / price

# -------------------------
# SIGNAL GENERATION
# -------------------------
def generate_signals():
    model = NBeats().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "nbeats.pt", map_location=DEVICE))
    model.eval()

    for d in TEST_DAYS:
        print(f"\nGenerating signals for day {d}...")
        prices = load_day_prices(d)
        n = len(prices)
        rows = []

        pbar = tqdm(range(n), desc=f"Day {d} ticks", ncols=120)
        for i in pbar:
            price = prices[i]

            if i < INPUT_WINDOW:
                rows.append([i, float(price), np.nan, np.nan, "NO_SIGNAL"])
                continue

            past = prices[i-INPUT_WINDOW:i].astype(np.float32)
            X = torch.from_numpy(past).unsqueeze(0).to(DEVICE)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                pred = model(X).cpu().numpy()[0]

            slope = linreg_slope(pred)
            pred_ret = slope_to_return(slope, price)

            if np.isnan(pred_ret):
                sig = "NO_SIGNAL"
            elif pred_ret >= LONG_THRESHOLD:
                sig = "LONG"
            elif pred_ret <= SHORT_THRESHOLD:
                sig = "SHORT"
            else:
                sig = "FLAT"

            rows.append([i, float(price), float(slope), float(pred_ret), sig])

        out_df = pd.DataFrame(rows, columns=["timestamp_idx", "price", "pred_slope", "pred_return", "signal"])
        out_file = SIGNAL_DIR / f"day{d}_signals.csv"
        out_df.to_csv(out_file, index=False)
        print(f"Saved signals → {out_file}")

# -------------------------
# ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise SystemExit(f"DATA_DIR missing: {DATA_DIR}")

    train_model()
    generate_signals()
