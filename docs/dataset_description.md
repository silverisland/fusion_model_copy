# Dataset Description

数据存储为 DataFrame。当前 `UnifiedDataset` 会把除时间戳外的每一列预先 stack 成
NumPy array，并在 `collate_fn` 中转成 batch 字典。

## Core Columns

| Column | Meaning | Expected Per-Sample Shape |
| --- | --- | --- |
| `timestamp_win` | 窗口时间戳，15 分钟分辨率 | scalar/timestamp |
| `observe_power` | 历史 7 天光伏功率 | `(672,)` |
| `observe_power_future` | 未来 2 天光伏功率，通常作为 target | `(192,)` |
| `GHI_solargis` | 历史 7 天 GHI | `(672,)` |
| `GHI_solargis_future` | 未来 2 天 GHI | `(192,)` |
| `TEMP_solargis` | 历史 7 天温度 | `(672,)` |
| `TEMP_solargis_future` | 未来 2 天温度 | `(192,)` |

## Batch Format

DataLoader 输出的 batch 是字典，主要字段 shape 为：

```text
observe_power:        (B, 672)
observe_power_future: (B, 192)
GHI_solargis:         (B, 672)
GHI_solargis_future:  (B, 192)
TEMP_solargis:        (B, 672)
TEMP_solargis_future: (B, 192)
index:                (B,)
column_names:         list[str]
```

如果 DataFrame 中存在额外数值列，数据集会自动加入 batch。模型应通过字段名访问数据。

## Target Convention

主训练入口 `run_longExp.py` 默认 `--target_key observe_power_future`。`base` 和
`expert_head` 默认使用 `observe_power_future`，`v4` 和 `v5` 会通过 factory 接收
`target_key`。新实验建议统一使用：

```text
target_key = observe_power_future
output shape = (B, n_features, pred_len)
```

如果 target 原始 shape 是 `(B, pred_len)`，融合模型内部通常会扩展成
`(B, 1, pred_len)`。

Compatibility note:

- `legacy` 和 `tensor_v3` 的训练路径仍硬编码读取 `batch["target_power"]`。
- 如果继续使用这些旧版本，需要在数据里提供 `target_power`，或先修改模型 target 读取逻辑。
