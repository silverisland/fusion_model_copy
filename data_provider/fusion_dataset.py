import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader  

class UnifiedDataset(Dataset):
    """
    Optimized Row-based dataset for fusion models. 
    Pre-converts DataFrame columns to NumPy arrays for fast indexing.
    """
    def __init__(self, df, timestamp_cols=['timestamp_win']):
        # 1. Identify and separate timestamp columns
        self.timestamp_cols = [col for col in timestamp_cols if col in df.columns]
        self.feature_cols = [col for col in df.columns if col not in self.timestamp_cols]
        self.timestamp = np.stack(df['timestamp_win'].values)

        # 2. Pre-process features into a dictionary of arrays
        self.data_dict = {}
        for col in self.feature_cols:
            vals = df[col].values
            # If the column contains numpy arrays (e.g. sequence data), stack them into a single (N, L) array
            if len(vals) > 0 and isinstance(vals[0], np.ndarray):
                self.data_dict[col] = np.stack(vals).astype(np.float32)
            else:
                self.data_dict[col] = vals.astype(np.float32)

        # 3. Also pre-process timestamp columns
        for col in self.timestamp_cols:
            vals = df[col].values
            self.data_dict[col] = vals

        self.column_names = self.feature_cols
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Fast indexing from dictionary of pre-stacked arrays (features + timestamps)
        sample = {col: self.data_dict[col][index] for col in self.feature_cols + self.timestamp_cols}
        sample['index'] = index
        return sample

def collate_fn(batch):
    """
    Optimized collate function.
    """
    feature_keys = [k for k in batch[0].keys() if k != 'index']
    collated = {}

    # Batching remains the same but inputs are now faster to access
    for key in feature_keys:
        # Using np.array for speed if not already numpy
        data_list = [b[key] for b in batch]
        first_elem = data_list[0] 
        if isinstance(first_elem, (str, np.str_, np.datetime64)):
            collated[key] = [pd.Timestamp(dt).to_pydatetime() for dt in data_list]
        else:
            collated[key] = torch.from_numpy(np.stack(data_list)).float()

    collated['index'] = torch.tensor([b['index'] for b in batch])
    collated['column_names'] = feature_keys

    return collated

def data_provider(args, data_df, flag):
    if flag == 'train':
        shuffle_flag = True 
        drop_last = False 
        batch_size = args.batch_size 
    elif flag == 'val':
        shuffle_flag = False 
        drop_last = True 
        batch_size = args.batch_size
    else:
        shuffle_flag = False # Test usually doesn't need shuffle
        drop_last = True 
        batch_size = args.batch_size 
    
    data_set = UnifiedDataset(
        df = data_df,  
    )

    data_loader = DataLoader(
        data_set, 
        batch_size = batch_size, 
        shuffle = shuffle_flag, 
        collate_fn = collate_fn, 
        drop_last = drop_last, 
    )

    return data_set, data_loader 