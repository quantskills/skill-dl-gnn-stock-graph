---
name: skill-dl-gnn-stock-graph
description: 当需要对A股市场进行GNN量化选股时，使用此skill。支持多层异构图（行业/概念/DTW/相关性）构建、GATs_ts与MF-IAMGCN双模型架构、五维特征工程、TopK选股策略、完整回测引擎（含T+1/涨跌停/手续费模拟）。
quantSkills:
  organization: https://github.com/quantskills
  repository: quantskills/skill-dl-gnn-stock-graph
  repository_url: https://github.com/quantskills/skill-dl-gnn-stock-graph
  project_type: skill
  collection: liangshuyuan-q44
  license: GPL-3.0
  category: deep-learning
  tags: [quant, gnn, deep-learning, stock-selection, a-stock, graph-attention, backtest, point-in-time, no-future-leak]
  platforms: [claude-code, codex, cursor, hermes, openclaw]
  language: zh-en
  status: active
  validation_level: runnable
  maintainer_type: community
  requires: ["pytorch>=2.0", "panda_data"]
  summary_zh: 基于多层异构图神经网络（GATs_ts / MF-IAMGCN）的A股量化选股，支持五维特征工程、TopK策略和完整回测引擎。
  summary_en: A-share quantitative stock selection using multi-layer heterogeneous GNNs (GATs_ts / MF-IAMGCN) with five-dimensional features, TopK strategy, and full backtest engine.
tags: [quant, gnn, deep-learning, stock-selection, a-stock, graph-attention, backtest]
---

```json qsh-form
{
  "version": 1,
  "task": {
    "placeholder": "例如：用 GATs_ts 从沪深300选30只股票",
    "required": true
  },
  "fields": [
    {
      "key": "date",
      "type": "date",
      "label": "扫描日期"
    },
    {
      "key": "model",
      "type": "select",
      "label": "模型",
      "options": [
        {"value": "gats_ts", "label": "GATs_ts (RNN + 动态图注意力)"},
        {"value": "mf_iamgcn", "label": "MF-IAMGCN (混频跨期注意力)"}
      ]
    },
    {
      "key": "index",
      "type": "select",
      "label": "股票池",
      "options": [
        {"value": "000300.SH", "label": "沪深300"},
        {"value": "000905.SH", "label": "中证500"},
        {"value": "000852.SH", "label": "中证1000"}
      ]
    },
    {
      "key": "top_k",
      "type": "number",
      "label": "选股数量"
    },
    {
      "key": "epochs",
      "type": "number",
      "label": "训练轮数（默认100）"
    }
  ],
  "prompt_template": "{{task}}；扫描日：{{date}}；模型：{{model}}；股票池：{{index}}；选股数量：{{top_k}}。仅使用 date < {{date}} 的数据。附件：{{#attachments}}"
}
```

# 深度图神经网络 · A股量化选股

## 适用场景
- 每日盘后想基于GNN预测得分选出沪深300/中证500/中证1000中最有上涨潜力的K只股票
- 想利用多层异构图捕捉行业联动、概念轮动、价格形态相似性等隐性市场结构
- 想融合混频数据（日频量价 + 季频基本面 + 情绪文本）做统一建模
- 想对选股策略做严格回测（含T+1、涨跌停、手续费、滑点的真实A股约束）

## Core Workflow

1. Resolve stock universe from index weights (CSI300 / CSI500 / CSI1000) at the scan date T.
2. Filter out ST/*ST stocks, sub-new stocks (< 60 days listed), and suspended stocks.
3. Pull 5-dimensional features using 17 PandaData APIs, strictly `date < T`.
4. Build multi-layer heterogeneous graph with explicit (industry/concept/institutional) and implicit (DTW/Pearson) edges.
5. Train GNN model (GATs_ts or MF-IAMGCN) on the training window; per-symbol train/val split.
6. Predict scores for all universe stocks and select TopK.
7. Optionally run backtest with T+1, price limits, and transaction costs.

## 数据接口（panda_data）

| 接口 | 用途 | 关键字段 |
|---|---|---|
| `get_last_trade_date` | 解析最近交易日 | `date` |
| `get_prev_trade_date` | 计算训练窗口起点 | `date` |
| `get_trade_cal` | A股交易日历对齐 | `date, is_trading_day` |
| `get_index_weights` | 锁定指数成分股 | `index_symbol, date, stock_symbol, weight` |
| `get_factor` | 日频 OHLCV + turnover + market_cap | `date, symbol, open, close, high, low, volume, amount, turnover, market_cap` |
| `get_stock_daily_post` | 后复权 OHLCV + 涨跌停价 + 停牌标记 | `date, symbol, pre_close, limit_up, limit_down, trade_status, name` |
| `get_index_daily` | 基准指数日线 | `symbol, date, close, pre_close` |
| `get_industry_constituents` | 申万行业成分 | `industry_code, stock_symbol` |
| `get_concept_constituents` | 概念板块成分 | `concept_code, stock_symbol` |
| `get_stock_industry` | 个股→行业映射 | `symbol, industry_code` |
| `get_fina_reports` | 三大报表（资产负债表/利润表/现金流量表） | `date, symbol, report_type, ...` |
| `get_share_float` | 总市值/流通市值/上市天数 | `symbol, total_shares, float_shares, listed_date` |
| `get_lhb_list` | 龙虎榜 | `date, symbol, ...` |
| `get_block_trade` | 大宗交易 | `date, symbol, ...` |
| `get_top_holders` | 前十大股东 | `symbol, holder_name, holding_ratio` |
| `get_stock_status_change` | ST/*ST 状态过滤 | `symbol, status` |
| `get_macro_*` | 宏观指标（GDP/CPI/PMI等） | `date, value` |

字段详见 `references/need_used_api.md`。

## 术语约定

- **T日** = 扫描日/预测日，训练集严格 `date < T`
- **图节点** = 单只股票，节点特征 = 五维特征展平
- **显性边** = 同行业/同概念/同供应链/同机构持仓
- **隐性边** = DTW 收益率序列相似度 Top-K / Pearson 相关性阈值 / 格兰杰因果显著对
- **混频** = 日频量价（高频）+ 季频基本面（低频）通过 MIDAS 方案对齐

## 股票池（Universe）

**沪深300 / 中证500 / 中证1000**（默认 `--index 000300.SH`）：

- 取 T 日 `get_index_weights` 锁定成分股
- 自动过滤 ST/*ST（`get_stock_status_change`）
- 自动过滤上市不满 60 天次新股（`get_share_float.listed_date`）
- 停牌日通过 `trade_status != 0` 在窗口构建阶段剔除

```bash
python scripts/scan.py --index 000300.SH   # 沪深300
python scripts/scan.py --index 000905.SH   # 中证500
python scripts/scan.py --index 000852.SH   # 中证1000
```

## 特征工程（五维因子体系）

### 1. 量价因子（日频，20 日窗口）

| # | 名称 | 公式 | 说明 |
|---|------|------|------|
| 1 | `ret` | `close / pre_close - 1` | 日收益 |
| 2 | `log_vol` | `log(volume + 1)` | 对数成交量 |
| 3 | `amplitude` | `(high - low) / pre_close` | 振幅 |
| 4 | `turnover` | `turnover` | 换手率 |
| 5 | `gap` | `open / pre_close - 1` | 开盘跳空 |
| 6 | `dist_limit_up` | `(limit_up - close) / close` | 距涨停 |
| 7 | `dist_limit_down` | `(close - limit_down) / close` | 距跌停 |
| 8 | `excess_ret` | `ret - index_ret` | 相对基准超额收益 |
| 9 | `mom_5d` | `close / close_lag5 - 1` | 5日动量 |
| 10 | `mom_20d` | `close / close_lag20 - 1` | 20日动量 |
| 11 | `volatility_20d` | `std(ret, 20)` | 20日波动率 |
| 12 | `macd` | EMA12 - EMA26 | MACD |
| 13 | `rsi` | 100 - 100/(1+RS) | 14日RSI |
| 14 | `money_flow` | `amount / amount_lag5 - 1` | 5日资金流变化 |

### 2. 基本面因子（季频，TTM + 最新季度）

| # | 名称 | 来源 | 说明 |
|---|------|------|------|
| 1 | `pe_ttm` | 财报/市值 | 滚动市盈率 |
| 2 | `pb` | 财报/市值 | 市净率 |
| 3 | `roe_ttm` | 利润表/资产负债表 | ROE TTM |
| 4 | `market_cap` | `get_factor` | 总市值 |
| 5 | `revenue_growth_yoy` | 利润表 | 营收同比增速 |
| 6 | `profit_growth_yoy` | 利润表 | 净利润同比增速 |
| 7 | `debt_ratio` | 资产负债表 | 资产负债率 |

### 3. 情绪因子（日频，外部注入）

| # | 名称 | 来源 | 说明 |
|---|------|------|------|
| 1 | `news_sentiment` | 外部 LLM | 新闻情感得分 [-1, 1] |
| 2 | `lhb_flag` | `get_lhb_list` | 是否上龙虎榜 |
| 3 | `block_trade_premium` | `get_block_trade` | 大宗交易溢价率 |

### 4. 宏观因子（月/季频）

| # | 名称 | 来源 | 说明 |
|---|------|------|------|
| 1 | `gdp_yoy` | `get_macro_na` | GDP同比增速 |
| 2 | `cpi_yoy` | `get_macro_pi` | CPI同比 |
| 3 | `pmi` | `get_macro_ci` | 制造业PMI |
| 4 | `m2_yoy` | `get_macro_mb` | M2同比增速 |

### 5. 关系特征（图结构特征）

| # | 名称 | 说明 |
|---|------|------|
| 1 | `degree_centrality` | 图节点度数 |
| 2 | `pagerank` | PageRank 中心性 |
| 3 | `dtw_similarity_mean` | 平均 DTW 相似度 |
| 4 | `industry_excess_ret` | 同行业平均超额收益 |

- **样本 = 20 日 × 14 量价特征 = 280 维 + 7 基本面 + 3 情绪 + 4 宏观 + 4 关系 = 298 维**
- **标准化**：训练集按列 z-score；Winsorize 1%/99% 极值压缩
- **严格无未来函数**：所有特征仅使用 `date < T` 的数据

## 模型

### GATs_ts（默认）

**RNN + 动态图注意力网络**，兼顾时序和关系：

```
输入: (N, lookback, D_price)
  ├── RNN Encoder (GRU, 2层, hidden=64) → 时序编码 (N, 64)
  ├── GAT Conv (4-head, hidden=32) → 图结构编码 (N, 128)
  └── MLP Head [64, 32] → 收益率预测 (N, 1)
```

- 图结构来自显性（行业/概念）+ 隐性（DTW Top-20/相关性 >0.5）
- 文献效果：相对沪深300年化超额 **28.9%**，信息比率 2.94

### MF-IAMGCN

**混频跨期注意力多层图卷积**，支持 MIDAS 混频抽样：

```
日频分支: GCN×3 (64 dim) → 日频表示 (N, 64)
季频分支: MLP → MIDAS对齐 → 季频表示 (N, 64)
跨期注意力: 4-head Attention → 融合表示 (N, 64)
预测头: MLP → 收益率预测 (N, 1)
```

## 选股策略

| 策略 | 说明 |
|---|---|
| **TopK** | 每日基于 GNN 预测得分降序，选前 K 只 |
| **阈值筛选** | 仅预测得分 > `--min_score` 的股票纳入 |
| **排序学习（LTR）** | 直接优化股票收益率排名（v0.2 计划） |

## 回测系统

### 交易规则

- **T+1**：当日买入次日才能卖出
- **涨跌停限制**：±10%（主板）/ ±20%（创业板/科创板）无法买卖
- **停牌处理**：停牌期间无法交易
- **手续费**：佣金 0.03% + 印花税 0.1%（仅卖出）+ 滑点 0.1%

### 绩效指标

| 类别 | 指标 |
|---|---|
| 收益 | 年化收益率、累计收益率 |
| 风险 | 最大回撤、年化波动率 |
| 风险调整 | 夏普比率、索提诺比率、信息比率、卡玛比率 |
| 超额 | 相对基准（沪深300/中证500）超额收益 |
| 交易 | 年化换手率、胜率 |

## 风控

- **单票集中度**：单只股票仓位上限 5%
- **最大回撤硬止损**：-20%
- **流动性过滤**：换手率 < 0.1% 的股票不纳入

## 使用方式

```bash
# 认证 & 环境（首次）
conda activate pandaai
pip install -r requirements.txt
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...

# 字段自检
python -m scripts.data.loader --self-check --date 20260729

# 单日选股 —— GATs_ts，沪深300，Top30
python scripts/scan.py --date 20260729 --model gats_ts

# 单日选股 —— MF-IAMGCN，中证500，Top20
python scripts/scan.py --date 20260729 --model mf_iamgcn --index 000905.SH --top_k 20

# 完整回测模式
python scripts/scan.py \
    --start 20250101 \
    --end 20260729 \
    --model gats_ts \
    --index 000300.SH \
    --top_k 30 \
    --backtest

# 完整参数
python scripts/scan.py \
    --date 20260729 \
    --model gats_ts \
    --index 000300.SH \
    --lookback 20 \
    --train_days 252 \
    --top_k 30 \
    --epochs 100 \
    --batch_size 64 \
    --seed 42 \
    --config config/model_config.yaml

# 单元测试
pytest tests/ -v
```

## Output Contract

### `output/gnn_picks_YYYYMMDD.csv`

Every selected stock is included with the required fields below. A stock is considered selected when its `rank <= top_k`.

| 列 | 类型 | 约束 |
|---|---|---|
| `trade_date` | string (YYYYMMDD) | 扫描日期，非空 |
| `rank` | integer | 连续 1..K，无缺漏，非空 |
| `symbol` | string | A股代码含后缀 (.SH / .SZ)，每文件唯一，非空 |
| `name` | string | 股票名称，非空 |
| `score` | float | GNN 预测得分，有限、非 NaN，单调非递增 |
| `ret_T` | float | T 日收益率（信息列，若 T 为当日可为 NaN） |
| `sector` | string | 申万一级行业名，非空 |
| `market_cap` | float | T 日总市值，非空 |

### `output/gnn_picks_YYYYMMDD.md`

Markdown 报告必须包含：
1. **TopK 排名表**：含 symbol、name、score、sector、market_cap
2. **行业分布**：各申万一级行业数量和占比
3. **模型元信息**：模型名、seed、train_days、lookback、epochs、特征数、节点数、边数、图类型

若启用回测模式，额外章节：绩效指标（年化收益/夏普/索提诺/卡玛/信息比率/最大回撤）和交易统计（换手率/胜率）。

### Model Checkpoint (`output/{model}_{date}_model.pt`)

PyTorch `state_dict`，包含 `model_state_dict`、`optimizer_state_dict`、`epoch`、`val_loss`、`seed`、`model_name`。Checkpoint 是大二进制文件，不应提交 Git。

## Validation

```bash
pytest tests/ -v
python scripts/validate.py output/gnn_picks_YYYYMMDD.csv
node scripts/validate-qsh-form.mjs SKILL.md
```

CSV 校验器检查列存在性、rank 单调性、score 有限性、日期格式和 symbol 唯一性。MD 校验器检查必需章节。qsh-form 校验器确保输入契约合法。

## Safety Boundary

Credentials are read from environment variables (`PANDA_DATA_USERNAME`, `PANDA_DATA_PASSWORD`) and removed from the process environment after loading. Never place credentials, raw private data, model checkpoints, or generated full-market datasets in Git.

`.gitignore` excludes `output/`, `models/`, `graph_cache/`, `*.log`, `*.parquet`.

This Community Project is for research and education only. GNN predictions carry inherent uncertainty; selected stocks are not guaranteed to outperform. It is not an official or verified QUANTSKILLS product and does not constitute investment advice.

## A股特殊处理清单

- [x] 后复权处理（`get_stock_daily_post`）
- [x] 涨跌停零值占位处理
- [x] 停牌前向填充（`ffill`）
- [x] 交易日历对齐（`get_trade_cal`）
- [x] ST/*ST 过滤（`get_stock_status_change`）
- [x] 次新股过滤（上市 < 60 天）
- [x] T+1 交易规则合规
- [x] 涨跌停买卖限制
- [x] 真实交易成本（佣金+印花税+滑点）

## References

- `references/data_guide.md` — 28 特征到 API 到公式的完整字段口径
- `references/methodology.md` — 方法冻结（v0.1.0），fail-closed 原则，无未来函数约束
- `references/source_boundary.md` — 允许/禁止数据源，可复现性保证
- `references/need_used_api.md` — 17 个 PandaData API 入参/响应详情

## 验收要求

- **无未来函数**：训练集严格 `date < T`，`test_no_future_leak.py` 覆盖
- **单元测试全通过**：`pytest tests/` 无失败，总数 ≥ 30
- **字段自检通过**：`python -m scripts.data.loader --self-check --date <近期日>` 返回 0
- **端到端跑通**：至少一个真实日期能产出 CSV + MD，TopK 完整无 NaN
- **训练可复现**：相同 `--seed` 两次运行的 TopK 顺序完全一致
- **回测合规**：T+1、涨跌停、手续费正确模拟，换手率合理
- **文档一致**：本文件的特征公式与 `scripts/features/` 实现一致

## 已知局限

- 新闻情感需外部 LLM 注入（panda_data 无新闻文本接口）；v0.1 默认不启用情绪因子
- 供应链关系无直接数据源，v0.1 默认关闭显性供应链边
- 格兰杰因果检验计算量大（O(N²·T)），v0.1 默认关闭
- 仅支持日频调仓，不支持日内调仓
- 图规模限制：单次不超过 500 节点（沪深300 + 中证500 合计约800只，需分片训练）
- v0.2 计划：Learning-to-Rank 排序损失、贝叶斯 GNN 不确定性估计、LLM 情感集成
