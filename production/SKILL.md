---
name: skill-dl-gnn-stock-graph-production
description: Read versioned GNN stock selection results from output CSV/MD and model checkpoint files without retraining or recomputing features.
---

# Production Result

Read results from the `output/` directory. The primary output files are:

- `output/gnn_picks_YYYYMMDD.csv` — ranked stock picks with scores
- `output/gnn_picks_YYYYMMDD.md` — human-readable report with sector distribution and model diagnostics
- `output/{model}_{date}_model.pt` — trained model checkpoint (if persistence is enabled)

## CSV Output Contract

Required columns:

| Column | Type | Constraint |
|---|---|---|
| `trade_date` | string (YYYYMMDD) | Must match the scan date, non-null |
| `rank` | integer | Sequential 1..K with no gaps, non-null |
| `symbol` | string | A-share code with exchange suffix (.SH / .SZ), unique per file, non-null |
| `name` | string | Stock name, non-null |
| `score` | float | GNN predicted score, finite, non-NaN, monotonically non-increasing |
| `ret_T` | float | T-day return (informational, may be NaN if T is the current/latest day) |
| `sector` | string | Shenwan L1 industry name, non-null |
| `market_cap` | float | Total market capitalization at T, non-null |

Only `rank <= top_k` represents a selection. Scores are model predictions, not expected returns.

## Markdown Report Contract

The Markdown report must contain:

1. **TopK List Table**: ranked stocks with symbol, name, score, sector, market_cap
2. **Sector Distribution**: count and percentage per Shenwan L1 industry
3. **Model Metadata**: model name, seed, train_days, lookback, epochs, feature count, node count, edge count, graph types used

If backtest mode was enabled, additional sections:
4. **Performance Metrics**: annualized return, volatility, max drawdown, Sharpe, Sortino, Calmar, Information Ratio
5. **Trading Statistics**: annual turnover, win rate, average holding period

## Model Checkpoint

The `.pt` file is a PyTorch `state_dict` saved with `torch.save()`. Loading:

```python
import torch
checkpoint = torch.load("output/gats_ts_20260731_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
```

Checkpoint metadata keys: `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_loss`, `seed`, `model_name`, `train_days`, `lookback`, `feature_count`.

## Diagnostics

CSV files should pass `scripts/validate.py` contract checks:
```bash
python scripts/validate.py output/gnn_picks_YYYYMMDD.csv
```

The validator checks column presence, rank monotonicity, score finiteness, date format, and symbol uniqueness.

## Safety

- Model checkpoints are large binary artifacts — never commit to Git.
- CSV/MD output files are informational only and do not constitute investment advice.
- This result is for research and education purposes.
- GNN predictions carry inherent uncertainty; selected stocks are not guaranteed to outperform.
