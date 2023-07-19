import numpy as np
import torch
from torch.utils.data import Dataset

class FashionDataset(Dataset):
    def __init__(self, params, data):
        self.params = params
        self.token_ids = np.array(data)
        self.lengths = np.array([len(t) for t in data])

        self.check()
        # self.remove_long_sequences()
        # self.remove_empty_sequences()
        # self.remove_unknown_sequences()
        # self.print_statistics()

    def __len__(self):
        return len(self.lengths)
    
    def __getitem__(self, index):
        return (self.token_ids[index], self.lengths[index])

    def check(self):
        """
        Some sanity checks
        """
        assert len(self.token_ids) == len(self.lengths)
        assert all(self.lengths[i] == len(self.token_ids[i]) for i in range(len(self.lengths)))

    def batch_sequences(self, batch):
        """
        Do the padding and transform into torch.tensor.
        """
        token_ids = [t[0] for t in batch]
        lengths = [t[1] for t in batch]
        assert len(token_ids) == len(lengths)

        tk_t = torch.tensor(np.array(token_ids))  # (bs, max_seq_len_)
        lg_t = torch.tensor(np.array(lengths))  # (bs)
        return tk_t, lg_t

def round_batch(x: torch.tensor, lengths: torch.tensor):
    # number of sentences == 0 [8]
    bs1 = len(lengths)
    bs2 = 8 * (bs1 // 8)
    assert bs2 > 0 and bs2 % 8 == 0
    if bs1 != bs2:
        idx = torch.randperm(bs1)[:bs2]
        lengths = lengths[idx]
        slen = lengths.max().item()
        x = x[idx, :slen]
    else:
        idx = None
    # sequence length == 0 [8]
    ml1 = x.size(1)
    if ml1 % 8 != 0:
        pad = 8 - (ml1 % 8)
        ml2 = ml1 + pad
        pad_id = 50257
        padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
        x = torch.cat([x, padding_tensor], 1)
        assert x.size() == (bs2, ml2)

    assert x.size(0) % 8 == 0
    assert x.size(1) % 8 == 0
    return x, lengths


def prepare_batch_clm(batch, is_round_batch = False, device='cuda', device_type='cuda'):
    token_ids, lengths = batch
    if is_round_batch:
        token_ids, lengths = round_batch(x=token_ids, lengths=lengths)
    assert token_ids.size(0) == lengths.size(0)
    attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
    clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
    clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility
    # sanity checks
    assert 0 <= token_ids.min() <= token_ids.max() < 50261

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        token_ids = token_ids.pin_memory().to(device, non_blocking=True) 
        attn_mask = attn_mask.pin_memory().to(device, non_blocking=True) 
        clm_labels = clm_labels.pin_memory().to(device, non_blocking=True)
    else:
        token_ids = token_ids.to(device) 
        attn_mask = attn_mask.to(device) 
        clm_labels = clm_labels.to(device)
        
    return token_ids, attn_mask, clm_labels


