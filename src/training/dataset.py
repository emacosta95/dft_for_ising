from typing import List, Dict, Tuple
import numpy as np
import argparse
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchmetrics import R2Score
from tqdm.notebook import tqdm, trange
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt


class ScalableCorrelationDataset(Dataset):
    def __init__(self, ns: list, corrs: list):
        super().__init__()  # In python 3 this is enough

        self.ns = ns  # list of tensors batch,dim
        self.corrs = corrs  # list of tensors batch,dim,dim
        print(f"self.ns[i].shape={self.ns[0].shape}")

    def __len__(self):
        return self.ns[0].shape[0]

    def __getitem__(self, idx):
        return [(self.ns[i][idx], self.corrs[i][idx]) for i in range(len(self.ns))]
