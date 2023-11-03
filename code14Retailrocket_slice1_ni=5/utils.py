import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

    def __len__(self):
        return len(self.data[0])

def collate_fn(batch):
    inps, targets = [], []
    for (inp, target) in batch:
        inps.append(inp)
        targets.append(target)
    max_len = max((len(l) for l in inps))
    inps = list(map(lambda l:l+[0]*(max_len - len(l)), inps))
    return inps, targets