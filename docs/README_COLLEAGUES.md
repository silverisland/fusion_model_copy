# 专家模型改造指南 (Expert Model Modification Guide)

为了支持融合模型（Fusion Model）的构建，我们需要统一各个专家模型的输入接口和数据格式。本指南说明了如何改造您的模型，以便能够被融合模型无缝调用。

## 1. 核心变更概览

1.  **数据处理外移**：请将原先在 `Dataset` 类中进行的特定列选择、标准化（Scaling）等预处理操作移动到模型内部的 `forward_hidden` 方法中。
2.  **统一输入接口**：模型必须实现 `forward_hidden(batch_dict)` 方法。
3.  **独立模型类**：确保您的模型可以被独立初始化，不再依赖于特定的 `DataLoader` 逻辑。

## 2. 统一的数据格式 (UnifiedDataset)

融合模型将使用 `data_provider/fusion_dataset.py` 中的 `UnifiedDataset`。它提供一个包含以下内容的字典（Batch）：

-   `batch['x_raw']`: 形状为 `(B, L, D)` 的张量，包含所有原始数值特征（Float32）。
-   `batch['column_names']`: 一个包含 `D` 个字符串的列表，对应 `x_raw` 中每一列的标题（Column Titles）。
-   `batch['y_raw']`: 形状为 `(B, P, D)` 的目标序列。

## 3. 您的模型需要做的改造

### 实现 `forward_hidden(self, batch_dict)`

在您的模型类中，请实现此方法以输出隐向量（Embedding）。

```python
class YourExpertModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 保存您的特定列名，用于在 forward_hidden 中筛选
        self.target_columns = ['OT', 'HUFL'] 
        # 如果有预训练好的 Scaler，也请在初始化时加载
        self.scaler = ... 

    def forward_hidden(self, batch_dict):
        # 1. 获取原始数据和列名
        x_raw = batch_dict['x_raw']  # (B, L, Total_D)
        all_cols = batch_dict['column_names']
        
        # 2. 特色列筛选 (通过标题匹配索引)
        indices = [all_cols.index(c) for c in self.target_columns if c in all_cols]
        x_input = x_raw[:, :, indices]
        
        # 3. 内部预处理 (标准化等)
        # x_input = (x_input - self.mean) / self.std
        
        # 4. 模型前向传播获取隐层输出
        # hidden = self.backbone(x_input)
        return hidden  # 返回 (B, L, d_model) 或 (B, d_model)
```

## 4. 如何验证

1.  请参考项目根目录下的 `demo.py`。
2.  在 `demo.py` 中实例化您的模型。
3.  运行 `demo.py`，确保您的模型能够成功处理 `UnifiedDataset` 产生的 batch 并输出预期的隐向量。

## 5. 提交要求

请将您的模型文件放入 `models/` 文件夹下（例如 `models/your_model_name.py`），并确保所有依赖项都在模型类内部处理。
