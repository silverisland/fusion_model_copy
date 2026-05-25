# Expert Model Integration Guide

本文档说明专家模型如何接入当前融合训练框架。当前项目不再使用早期的
`x_raw/y_raw` 统一张量接口，而是直接把 DataFrame 的每一列作为 batch 字段传给
专家模型。

## 1. Current Batch Format

数据由 `data_provider/fusion_dataset.py` 读取。每个 batch 是一个字典，常用字段包括：

| Key | Shape | Description |
| --- | --- | --- |
| `observe_power` | `(B, 672)` | 历史 7 天光伏功率，15 分钟分辨率。 |
| `observe_power_future` | `(B, 192)` | 未来 2 天光伏功率，通常作为训练 target。 |
| `GHI_solargis` | `(B, 672)` | 历史 7 天 GHI。 |
| `GHI_solargis_future` | `(B, 192)` | 未来 2 天 GHI。 |
| `TEMP_solargis` | `(B, 672)` | 历史 7 天温度。 |
| `TEMP_solargis_future` | `(B, 192)` | 未来 2 天温度。 |
| `timestamp_win` | list-like | 窗口时间戳。 |
| `index` | `(B,)` | 样本索引。 |
| `column_names` | list[str] | 当前 batch 中除 `index` 外的字段名。 |

如果 DataFrame 中增加新列，`UnifiedDataset` 会自动把它加入 batch。专家模型应通过
字段名读取自己需要的数据，而不是依赖固定列顺序。

## 2. Required Expert Interface

每个专家模型必须实现：

```python
def forward_hidden(self, batch: dict):
    ...
```

融合模型会冻结专家参数，并通过 `forward_hidden(batch)` 取出 hidden state：

```python
for name, model in self.expert_models.items():
    model.eval()
    with torch.no_grad():
        batch_tensor[name] = model.forward_hidden(batch)
```

返回值可以是以下常见形式：

```text
(B, D)
(B, T, D)
(B, C, T, D)
```

具体 shape 需要和融合模型里的 `--fusion_expert_dims` 或默认专家维度一致。当前默认：

```text
m1: 512
m2: 256
m3: 384
m4: 512
```

## 3. Prediction Head Reconstruction Requirement

如果要做 `expert_head` 实验，`forward_hidden(batch)` 必须返回原专家预测头真实接收的
hidden tensor。否则即使预测头结构复原正确，也无法复现专家原始精度。

最低 sanity check：

```text
original expert input -> original expert head -> original prediction
forward_hidden(batch) -> copied original head -> same prediction
```

如果这个检查失败，优先排查：

- `forward_hidden` 返回的是否是原 head 输入，而不是中间层或池化后的替代表示。
- 专家内部的标准化、反标准化、时间特征处理是否一致。
- 输出 shape 是否最终统一为 `(B, n_features, pred_len)`。
- target 是否使用 `observe_power_future`，或显式设置了正确的 `--target_key`。
- 如果跑 `legacy` 或 `tensor_v3`，当前代码仍需要 `batch["target_power"]`。

## 4. Implementation Notes

专家模型内部负责处理自己的输入字段、归一化和特征构造。例如：

```python
class YourExpertModel(nn.Module):
    def forward_hidden(self, batch):
        power = batch["observe_power"]
        ghi = batch["GHI_solargis"]
        ghi_future = batch["GHI_solargis_future"]

        # Apply the same preprocessing used by the original expert.
        x = self.build_inputs(power, ghi, ghi_future)
        hidden = self.backbone_forward_hidden(x)
        return hidden
```

不要在融合数据集里为某个专家写专用预处理逻辑。这样会让多个专家的数据路径难以对齐，
也会让融合实验难以复现。

## 5. Validation Commands

Syntax check:

```bash
.pixi/envs/default/bin/python -m py_compile run_longExp.py exp/*.py models/**/*.py utils/*.py
```

Registry check:

```bash
.pixi/envs/default/bin/python -c "from models.factory import fusion_version_choices; print(fusion_version_choices())"
```

Single expert-head smoke command:

```bash
.pixi/envs/default/bin/python run_longExp.py --is_training 1 --model_id head_m1 --model FusionModel --fusion_version expert_head --fusion_expert_name m1 --data custom
```
