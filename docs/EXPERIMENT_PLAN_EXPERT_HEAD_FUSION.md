# Expert Head Fusion Experiment Plan

## 1. Goal

The current fusion models do not outperform simple mean ensemble. This plan is
designed to isolate the cause:

```text
original expert predictions
-> expert hidden states + reconstructed prediction heads
-> multi-head joint training
-> prediction-level fusion
-> hidden-assisted prediction fusion
```

The key question is whether the fusion model is weak, or whether the current
framework loses expert capability before fusion begins.

## 2. Stage 0: Fixed Baselines

Use the same train/validation/test split, metrics, target shape, and
normalization or denormalization path for all experiments.

| ID | Experiment | Purpose |
| --- | --- | --- |
| B0 | Original prediction from each expert: m1, m2, m3, m4 | Measure real single-expert performance. |
| B1 | Simple mean of original expert predictions | Main baseline to beat. |
| B2 | Best single expert | Check how much mean ensemble improves over one expert. |
| B3 | Oracle best expert per sample | Estimate the upper bound of expert selection. |

Record:

- MAE, MSE, RMSE.
- Error by forecast horizon.
- Error by daytime/nighttime if applicable.
- Error by high-volatility and low-volatility samples.

Interpretation:

- If B1 barely improves over B2, expert diversity may be weak.
- If B3 is much better than B1, there is useful expert complementarity and a
  learned fusion model may help.

## 3. Stage 1: Single Expert Head Reconstruction

Freeze each expert backbone, take `forward_hidden(batch)`, and train only a
reconstructed prediction head.

| ID | Experiment | Example Setting |
| --- | --- | --- |
| H1-m1 | Reconstruct and train m1 head | `--fusion_version expert_head --fusion_expert_name m1` |
| H1-m2 | Reconstruct and train m2 head | `--fusion_version expert_head --fusion_expert_name m2` |
| H1-m3 | Reconstruct and train m3 head | `--fusion_version expert_head --fusion_expert_name m3` |
| H1-m4 | Reconstruct and train m4 head | `--fusion_version expert_head --fusion_expert_name m4` |

Important sanity checks:

| ID | Experiment | Purpose |
| --- | --- | --- |
| H0 | Copy the original expert head weights and test without training | Verify that `forward_hidden + head` reproduces the expert output. |
| H0-ft | Copy original head weights, then fine-tune | Check whether the local training framework can preserve or improve expert performance. |

Pass criteria:

```text
reconstructed single-expert head ~= original expert performance
```

If this fails, do not move to complex fusion yet. First check:

- Whether `forward_hidden(batch)` is exactly the tensor consumed by the original
  prediction head.
- Whether the original head architecture was faithfully reconstructed.
- Whether target shape is consistently `(B, n_features, pred_len)`.
- Whether normalization and denormalization are identical to expert training.
- Whether loss, data split, and evaluation metrics match the original expert.

## 4. Stage 2: Joint Training of Multiple Expert Heads

Train multiple reconstructed heads at the same time, but do not fuse their
outputs yet.

| ID | Experiment | Output |
| --- | --- | --- |
| MH1 | Joint train m1 + m2 heads | `pred_m1`, `pred_m2` |
| MH2 | Joint train m1 + m2 + m3 heads | `pred_m1`, `pred_m2`, `pred_m3` |
| MH3 | Joint train m1 + m2 + m3 + m4 heads | `pred_m1`, `pred_m2`, `pred_m3`, `pred_m4` |
| MH4 | Shared adapter + expert-specific heads | Same as above |

Start with this loss:

```text
loss = mean(loss_m1, loss_m2, loss_m3, loss_m4)
```

Interpretation:

- Joint heads ~= separately trained heads: no obvious multi-task interference.
- Joint heads worse: check loss scale, learning rate, dropout/batch norm, and
  shared parameters.
- Joint heads better: shared representation may be useful for later fusion.

## 5. Stage 3: Prediction-Level Fusion

First try to beat simple mean in prediction space before returning to complex
hidden fusion.

| ID | Fusion Method | Description |
| --- | --- | --- |
| F1 | Simple mean | Reproduce the baseline. |
| F2 | Learned static weights | Learn global `softmax(w)` over experts. |
| F3 | Horizon-wise weights | Learn one expert weight vector for each forecast step. |
| F4 | Feature/horizon-wise weights | Learn weights per feature and forecast step. |
| F5 | Sample-dependent gate | Generate expert weights from history or hidden states. |
| F6 | Residual correction | Predict `mean_pred + correction_net(...)`. |

Recommended order:

```text
F2 -> F3 -> F5 -> F6
```

Do not start with a large MoE. A small learned prediction-level fusion model is
the cleanest proof that learnable fusion can beat fixed averaging.

## 6. Stage 4: Hidden-Assisted Fusion

Use hidden states to help prediction fusion, instead of immediately asking
hidden fusion to produce the final forecast from scratch.

| ID | Experiment | Description |
| --- | --- | --- |
| HF1 | Hidden states only generate gate weights | Final output is still weighted expert predictions. |
| HF2 | Hidden states generate residual correction | Correct mean or weighted-mean prediction. |
| HF3 | Hidden fusion directly outputs forecast | Highest risk; use only after earlier checks pass. |
| HF4 | Prediction fusion + hidden fusion two-branch model | Candidate final model. |

Preferred direction:

```text
expert predictions -> learned gate
expert hidden states -> gate or residual features
```

This is easier to debug and interpret than mixing all hidden states into a
single latent representation and directly predicting power.

## 7. Analysis Required for Each Stage

Save these artifacts whenever possible:

- Single-expert error distribution.
- Mean ensemble error distribution.
- Per-horizon error curves.
- Learned expert weights.
- Horizon-wise expert weights.
- Grouped metrics for daytime/nighttime and high/low volatility.
- Comparison against simple mean with the same data split.

These analyses are needed to prove that learned fusion is using expert
complementarity rather than adding complexity without benefit.

## 8. Recommended Execution Order

```text
1. Run B0/B1/B2/B3 and lock the baseline table.
2. Run H0 by copying original expert head weights.
3. Run H1 for each expert with reconstructed heads.
4. Run MH3 for all four expert heads without fusion.
5. Run F2 and F3 for static and horizon-wise prediction fusion.
6. Run F5 for sample-dependent gating.
7. Run F6 or HF2 for residual correction.
8. Compare the best result with existing v4/v5/tensor_v3 hidden fusion models.
```

## 9. Checkpoints

Checkpoint 1:

```text
single reconstructed expert head ~= original expert performance
```

Checkpoint 2:

```text
mean of reconstructed heads ~= mean of original expert predictions
```

Checkpoint 3:

```text
learned prediction fusion > simple mean ensemble
```

If Checkpoint 1 fails, focus on expert head reconstruction. If Checkpoint 2
fails, inspect joint training and reconstruction quality. If Checkpoint 3 fails,
the experts may not provide enough complementary information, or the gate input
does not contain the information needed to choose among them.
