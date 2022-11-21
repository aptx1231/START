import os
import torch
import pickle
import pandas as pd
from logging import getLogger
from libcity.data.dataset import BaseDataset, padding_mask, TrajectoryProcessingDataset
from libcity.data.dataset.bertlm_dataset import noise_mask


class ContrastiveSplitDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.argu1 = config.get('out_data_argument1', None)
        self.argu2 = config.get('out_data_argument2', None)
        self.data_argument1 = self.config.get("data_argument1", [])
        self.data_argument2 = self.config.get("data_argument2", [])
        self.masking_ratio = self.config.get('masking_ratio', 0.2)
        self.masking_mode = self.config.get('masking_mode', 'together')
        self.distribution = self.config.get('distribution', 'random')
        self.avg_mask_len = self.config.get('avg_mask_len', 3)
        self.collate_fn = collate_unsuperv_contrastive_split

    def _gen_dataset(self):
        train_dataset = TrajectoryProcessingDatasetSplit(
            data_name=self.dataset, data_type='train', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=self.max_train_size,
            argu1=self.argu1, argu2=self.argu2,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio,
            masking_mode=self.masking_mode,
            distribution=self.distribution,
            avg_mask_len=self.avg_mask_len)
        eval_dataset = TrajectoryProcessingDatasetSplit(
            data_name=self.dataset, data_type='eval', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=None,
            argu1=self.argu1, argu2=self.argu2,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio,
            masking_mode=self.masking_mode,
            distribution=self.distribution,
            avg_mask_len=self.avg_mask_len)
        test_dataset = TrajectoryProcessingDatasetSplit(
            data_name=self.dataset, data_type='test', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=None,
            argu1=self.argu1, argu2=self.argu2,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio,
            masking_mode=self.masking_mode,
            distribution=self.distribution,
            avg_mask_len=self.avg_mask_len)
        return train_dataset, eval_dataset, test_dataset


class TrajectoryProcessingDatasetSplit(TrajectoryProcessingDataset):

    def __init__(self, data_name, data_type, vocab, seq_len=512, add_cls=True,
                 merge=True, min_freq=1, max_train_size=None, argu1=None, argu2=None,
                 data_argument1=None, data_argument2=None,
                 masking_ratio=0.2, masking_mode='together',
                 distribution='random', avg_mask_len=3):

        self.vocab = vocab
        self.seq_len = seq_len
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self._logger = getLogger()
        self._logger.info('Init TrajectoryProcessingDatasetSplit!')

        self.masking_ratio = masking_ratio
        self.masking_mode = masking_mode
        self.distribution = distribution
        self.avg_mask_len = avg_mask_len
        self.exclude_feats = None
        self.data_argument1 = data_argument1
        self.data_argument2 = data_argument2
        if 'mask' in self.data_argument1:
            self._logger.info('Use mask as data argument in view1!')
        if 'mask' in self.data_argument2:
            self._logger.info('Use mask as data argument in view2!')

        if argu1 is not None:
            self.data_path1 = 'raw_data/{}/{}_{}_enhancedby{}.csv'.format(
                data_name, data_name, data_type, argu1)
            self.cache_path1 = 'raw_data/{}/cache_{}_{}_{}_{}_{}_enhancedby{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq, argu1)
        else:
            self.data_path1 = 'raw_data/{}/{}_{}.csv'.format(
                data_name, data_name, data_type)
            self.cache_path1 = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq)
        if argu2 is not None:
            self.data_path2 = 'raw_data/{}/{}_{}_enhancedby{}.csv'.format(
                data_name, data_name, data_type, argu2)
            self.cache_path2 = 'raw_data/{}/cache_{}_{}_{}_{}_{}_enhancedby{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq, argu2)
        else:
            self.data_path2 = 'raw_data/{}/{}_{}.csv'.format(
                data_name, data_name, data_type)
            self.cache_path2 = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq)

        self.temporal_mat_path1 = self.cache_path1[:-4] + '_temporal_mat.pkl'
        self.temporal_mat_path2 = self.cache_path2[:-4] + '_temporal_mat.pkl'
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.cache_path1) and os.path.exists(self.temporal_mat_path1) \
                and os.path.exists(self.cache_path2) and os.path.exists(self.temporal_mat_path2):
            self.traj_list1 = pickle.load(open(self.cache_path1, 'rb'))
            self.temporal_mat_list1 = pickle.load(open(self.temporal_mat_path1, 'rb'))
            self.traj_list2 = pickle.load(open(self.cache_path2, 'rb'))
            self.temporal_mat_list2 = pickle.load(open(self.temporal_mat_path2, 'rb'))
            self._logger.info('Load dataset from {}, {}'.format(self.cache_path1, self.cache_path2))
        else:
            origin_data_df1 = pd.read_csv(self.data_path1, sep=';')
            origin_data_df2 = pd.read_csv(self.data_path2, sep=';')
            assert origin_data_df1.shape == origin_data_df2.shape
            self.traj_list1, self.temporal_mat_list1 = self.data_processing(
                origin_data_df1, self.data_path1, cache_path=self.cache_path1, tmat_path=self.temporal_mat_path1)
            self.traj_list2, self.temporal_mat_list2 = self.data_processing(
                origin_data_df2, self.data_path2, cache_path=self.cache_path2, tmat_path=self.temporal_mat_path2)
        if self.max_train_size is not None:
            self.traj_list1 = self.traj_list1[:self.max_train_size]
            self.temporal_mat_list1 = self.temporal_mat_list1[:self.max_train_size]
            self.traj_list2 = self.traj_list2[:self.max_train_size]
            self.temporal_mat_list2 = self.temporal_mat_list2[:self.max_train_size]

    def __len__(self):
        assert len(self.traj_list1) == len(self.traj_list2)
        return len(self.traj_list1)

    def __getitem__(self, ind):
        traj_ind1 = self.traj_list1[ind]  # (seq_length, feat_dim)
        traj_ind2 = self.traj_list2[ind]  # (seq_length, feat_dim)
        temporal_mat1 = self.temporal_mat_list1[ind]  # (seq_length, seq_length)
        temporal_mat2 = self.temporal_mat_list2[ind]  # (seq_length, seq_length)
        mask1 = None
        mask2 = None
        if 'mask' in self.data_argument1:
            mask1 = noise_mask(traj_ind1, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        if 'mask' in self.data_argument2:
            mask2 = noise_mask(traj_ind2, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        return torch.LongTensor(traj_ind1), torch.LongTensor(traj_ind2), \
               torch.LongTensor(temporal_mat1), torch.LongTensor(temporal_mat2), \
               torch.LongTensor(mask1) if mask1 is not None else None, \
               torch.LongTensor(mask2) if mask2 is not None else None


def _inner_slove_data(features, temporal_mat, batch_size, max_len, vocab=None, mask=None):
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
        if mask[i] is not None:
            target_masks[i, :end, :] = mask[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)

    if mask[0] is not None:
        X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
        X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    return X, batch_temporal_mat, padding_masks

def collate_unsuperv_contrastive_split(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features1, features2, temporal_mat1, temporal_mat2, mask1, mask2 = zip(*data)  # list of (seq_length, feat_dim)
    X1, batch_temporal_mat1, padding_masks1 = _inner_slove_data(
        features1, temporal_mat1, batch_size, max_len, vocab, mask1)
    X2, batch_temporal_mat2, padding_masks2 = _inner_slove_data(
        features2, temporal_mat2, batch_size, max_len, vocab, mask2)
    return X1.long(), X2.long(), padding_masks1, padding_masks2, \
           batch_temporal_mat1.long(), batch_temporal_mat2.long()
