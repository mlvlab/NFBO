import numpy as np
import pandas as pd
import os
import torch
from data.scaler import NScaler, MinMaxScaler, MinMax11Scaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from omegaconf import OmegaConf
import torch.nn.functional as F


def get_default(value, default):
    """Return the value if it is not None, otherwise return the default."""
    if value is not None:
        return value
    else:
        return default

class OracleDataset(Dataset):
    def __init__(self, x, y, sclaer):
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)
        super().__init__()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DataStorage:
    def __init__(self, verbose, root_dir, batch_size, obj_func, data_type, scaler_type, graph_data_name, oracle_data_name, pretrain_data_name, oracle_data_load_num, train_ratio, val_ratio, **kwargs):
        cfg = dict(locals()); cfg.pop('self')
        self.__dict__.update(cfg)
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg

    def load_oracle_data(self):
        """Load oracle data.
        
        If oracle_data_load_num is -1, all data will be loaded.
        """
        data_dir = os.path.join(self.root_dir, f"data/{self.data_type}/oracle")
        if self.data_type == 'guacamol':
            data_path = os.path.join(data_dir, f"{self.oracle_data_name}")
            df = pd.read_csv(data_path)
            train_x_smiles = df['smile'].values
            train_x_selfies = df['selfie'].values 
            train_y = torch.from_numpy(df[self.cfg.kwargs.task_id].values).float() 
            train_y = train_y.unsqueeze(-1)
            datas = [train_x_smiles, train_x_selfies, train_y]
            
            #####################################
            # # valt 500 #
            # idx = torch.where(train_y>0)[0]
            # train_y[0:2] = train_y[idx]
            # train_x_smiles[0:2] = train_x_smiles[idx]
            # train_x_selfies[0:2] = train_x_selfies[idx]
            #####################################
            
            if self.pretrain_data_name is not None:
                data_path = os.path.join(data_dir, f"{self.pretrain_data_name}")
                df = pd.read_csv(data_path)
                train_x_smiles = df['smile'].values
                train_x_selfies = df['selfie'].values
            
            return datas if self.oracle_data_load_num == -1 else tuple(map(lambda x: x[:self.oracle_data_load_num], datas)) + (train_x_smiles, train_x_selfies)
        elif self.data_type == 'ZINC':
            data_path = os.path.join(data_dir, f"{self.oracle_data_name}")
            df = pd.read_csv(data_path)
            train_x_smiles = df['smile'].values
            train_x_selfies = df['selfie'].values 
            train_y = torch.full((len(train_x_smiles), 1), -1).float()
            datas = [train_x_smiles, train_x_selfies, train_y]
            
            if self.pretrain_data_name is not None:
                data_path = os.path.join(data_dir, f"{self.pretrain_data_name}")
                df = pd.read_csv(data_path)
                train_x_smiles = df['smile'].values
                train_x_selfies = df['selfie'].values
            
            return datas if self.oracle_data_load_num == -1 else tuple(map(lambda x: x[:self.oracle_data_load_num], datas)) + (train_x_smiles, train_x_selfies)
        
    def load_graph_data(self):
        data_dir = os.path.join(self.root_dir, f"data/{self.data_type}/graph/{self.graph_data_name}")
        raise ValueError("self.data_type muset be one of [].")
        
    def make_dataloader(self, objective = None):
        if self.scaler_type == "z":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type is None:
            self.scaler = NScaler()
        else:
            raise ValueError("scaler_type must be one of ['z', 'minmax', None].")
        
        def collate_fn(data):
            # Length of longest molecule in batch 
            max_size = max([x.shape[-1] for x in data])
            return torch.vstack(
                # Pad with stop token
                [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
            )
        if self.data_type == 'guacamol':
            x_smiles, x_selfies, self.y, pretrain_x_smiles, pretrain_x_selfies = self.load_oracle_data()
            x_token, pretrain_x_token = [objective.dataobj.tokenize_selfies(xx) for xx in [x_selfies, pretrain_x_selfies]]
            #temp = pretrain_x_smiles;pretrain_x_selfies=pretrain_x_selfies[0:100];x_selfies=x_selfies[0:100];self.y=self.y[0:100]
            self.x, self.pretrain_x = [collate_fn([objective.dataobj.encode(x) for x in x_token]) for x_token in [x_token, pretrain_x_token]]
            pass
            self.edge_index = None
            self.edge_weight = None
            self.node_feature = None
            self.graph = None
        elif self.data_type == 'ZINC':
            x_smiles, x_selfies, self.y, pretrain_x_smiles, pretrain_x_selfies = self.load_oracle_data()
            x_token, pretrain_x_token = [objective.dataobj.tokenize_selfies(xx) for xx in [x_selfies, pretrain_x_selfies]]
            self.x, self.pretrain_x = [collate_fn([objective.dataobj.encode(x) for x in x_token]) for x_token in [x_token, pretrain_x_token]]
            pass
            self.edge_index = None
            self.edge_weight = None
            self.node_feature = None
            self.graph = None
        else:
            raise ValueError("self.data_type muse be one of [].")
        
        if self.verbose > 1: print(f"Load len(x)={len(self.x)}, len(y)={len(self.y)}")
        assert len(self.x) == len(self.y), f"Input length {len(self.x)} and output length {len(self.y)} are must be same."
        
        data_num = len(self.x)
        train_num = int(data_num * self.train_ratio)
        val_num = int(data_num * self.val_ratio)
        test_num = data_num - train_num - val_num
        
        rp = torch.randperm(data_num)
        train_idx = rp[:train_num]
        val_idx = rp[train_num:train_num+val_num]
        test_idx = rp[train_num+val_num:]
        
        self.train_x, self.train_y = self.x[train_idx], self.y[train_idx]
        self.val_x, self.val_y = self.x[val_idx], self.y[val_idx]
        self.test_x, self.test_y = self.x[test_idx], self.y[test_idx]
        
        if self.verbose: print(f"Load len(train_x)={len(self.train_x)}, len(train_y)={len(self.train_y)}")
        if self.verbose: print(f"Load len(val_x)={len(self.val_x)}, len(val_y)={len(self.val_y)}")
        if self.verbose: print(f"Load len(test_x)={len(self.test_x)}, len(test_y)={len(self.test_y)}")
        
        self.scaler.fit(self.train_y)
        
        self.all_loader = DataLoader(OracleDataset(self.x, self.y, self.scaler), batch_size=self.batch_size, shuffle=False)
        self.train_loader = DataLoader(OracleDataset(self.train_x, self.train_y, self.scaler), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(OracleDataset(self.val_x, self.val_y, self.scaler), batch_size=self.batch_size, shuffle=False)
        if test_num <= 0:
            self.test_loader = None
        else:
            self.test_loader = DataLoader(OracleDataset(self.test_x, self.test_y, self.scaler), batch_size=self.batch_size, shuffle=False)
        
        return self.graph, self.x, self.y, self.all_loader, self.train_loader, self.val_loader, self.test_loader, self.scaler, self.pretrain_x
    