import numpy as np
import torch


def top_k(loc_pred, loc_true, topk):
    loc_pred = torch.FloatTensor(loc_pred)  # (batch_size * output_dim)
    val, index = torch.topk(loc_pred, topk, 1)
    index = index.numpy()  # (batch_size * topk)
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):  # i->batch, p->(topk,)
        target = loc_true[i]
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            # rank_index is start from 0, so need plus 1
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg
