# Fusion Model Reflection

本文档记录当前融合模型的主要风险和后续改进方向。它替代早期关于
`PackerMoEFusion` 的反思版本，重点对齐当前代码状态。

## 1. Current Implementation Status

当前融合版本由 `models/factory.py` 注册：

```text
base
expert_head
legacy
v2
v3
v4
v5
tensor_v3
```

其中 `base`、`expert_head`、`v4`、`v5`、`tensor_v3` 属于 hidden-only fusion 路线，
会通过 `FusionModelWithExperts` 冻结专家模型并调用各专家的 `forward_hidden(batch)`。

当前最重要的事实：

- `expert_head` 已有实验入口，但各专家预测头仍是占位实现，需要替换成真实专家 head。
- `v2`、`legacy`、`tensor_legacy`、`tensor_v3` 已经使用 `Conv1d` 或类似聚合逻辑，不应再简单描述为全部存在 `view + slice` 截断问题。
- `v4`、`v5` 已经引入 query-weighted forecast head 和 hidden gate，但仍需要和简单均值、线性融合、预测级 gate 做严格对比。
- `legacy` 和 `tensor_v3` 的训练路径仍硬编码 `batch["target_power"]`，和当前默认 `observe_power_future` 不完全一致。

## 2. Main Risk: Hidden Reconstruction May Be the Bottleneck

当前融合模型输给简单均值，并不必然说明融合思想错误。简单均值使用的是专家完整预测输出；
hidden fusion 则通常需要从冻结专家 hidden 中重新学习预测能力。

因此必须先回答：

```text
forward_hidden(batch) + reconstructed original head
是否能复现专家原始预测精度？
```

如果不能复现，后续复杂 attention、MoE、gate 的结果都很难解释。

## 3. Remaining Technical Risks

### 3.1 Expert Hidden Interface

不同专家的 hidden state 可能不是同一语义层级，甚至可能不是原预测头真实输入。直接拼接、
投影或 cross-attention 可能会丢失专家原有能力。

Required checks:

- 每个专家的 `forward_hidden(batch)` 是否返回原 head 输入。
- hidden shape 是否和 `fusion_expert_dims` 匹配。
- 专家内部归一化和时间特征处理是否完整保留。

### 3.2 Fusion Baseline Is Not Strong Enough

复杂 hidden fusion 必须先赢过以下简单基线：

```text
simple mean
learned static weights
horizon-wise weights
sample-dependent prediction gate
mean prediction + residual correction
```

如果这些预测级融合都无法稳定超过均值，说明专家互补性可能不足，或者 gate 输入缺少选择专家所需的信息。

### 3.3 MoE Justification

MoE 需要证明它学到了有意义的分工，而不是只增加参数量。

Required evidence:

- expert weight 分布。
- 不同 horizon 的专家权重。
- 高波动/低波动、白天/夜晚分组下的权重变化。
- 去掉 MoE 后的 FFN baseline。

### 3.4 Interpretability

融合模型应该能说明“什么时候相信哪个专家”。如果所有 hidden 被混成一个不可追踪的 latent
representation，论文或报告里的解释力会很弱。

优先考虑：

```text
expert predictions -> final weighted prediction
expert hidden states -> gate/residual features
```

这比直接 hidden-to-forecast 更容易调试和解释。

## 4. Recommended Roadmap

近期路线以 `docs/EXPERIMENT_PLAN_EXPERT_HEAD_FUSION.md` 为准：

```text
1. 建立原始专家和简单均值基线。
2. 复原每个专家预测头。
3. 验证 copied original head 是否能复现专家输出。
4. 同时训练多个专家 head，但暂不融合。
5. 从 prediction-level fusion 开始超过 simple mean。
6. 再引入 hidden-assisted gate 或 residual correction。
7. 最后与 v4/v5/tensor_v3 hidden fusion 对比。
```

## 5. What Is No Longer Current

早期文档中提到的以下内容目前不应作为已实现能力描述：

- 完整 `PackerMoEFusion` 预训练框架。
- Contrastive alignment loss。
- 两阶段预训练策略。
- Dynamic RevIN 输出补偿。
- PEFT / router-only tuning。
- Elastic query resizing。

这些可以作为长期研究想法，但当前实验和代码说明应以实际实现为准。
