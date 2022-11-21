import torch
from libcity.data.dataset import BERTLMDataset
from libcity.data.dataset.bertlm_dataset import collate_unsuperv_mask


class ContrastiveLMDataset(BERTLMDataset):
    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_unsuperv_contrastive_lm


def collate_unsuperv_contrastive_lm(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, masks, temporal_mat = zip(*data)

    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)
    contra_view1 = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        contra_view1[i, :end, :] = features[i][:end, :]
    masked_x, targets, target_masks, padding_masks, batch_temporal_mat = collate_unsuperv_mask(
        data=data, max_len=max_len, vocab=vocab, add_cls=add_cls)
    return contra_view1.long(), contra_view1.long().clone(), masked_x, targets, target_masks, padding_masks, batch_temporal_mat
