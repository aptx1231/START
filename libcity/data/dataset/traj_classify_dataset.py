import numpy as np
from logging import getLogger
from libcity.data.dataset import BaseDataset, TrajectoryProcessingDataset, padding_mask
from tqdm import tqdm
import torch
import datetime
import pickle


class TrajClassifyDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.classify_label = self.config.get('classify_label', 'vflag')
        if self.classify_label == 'vflag':
            self.collate_fn = collate_superv_classify_vflag
        elif self.classify_label == 'usrid':
            self.collate_fn = collate_superv_classify_usrid
        else:
            raise ValueError('Error classify_label = {}'.format(self.classify_label))

    def _gen_dataset(self):
        train_dataset = ClassifySubDataset(data_name=self.dataset, data_type='train',
                                           vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                           merge=self.merge, min_freq=self.min_freq,
                                           max_train_size=self.max_train_size,
                                           classify_label=self.classify_label)
        eval_dataset = ClassifySubDataset(data_name=self.dataset, data_type='eval',
                                          vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                          merge=self.merge, min_freq=self.min_freq,
                                          max_train_size=None,
                                          classify_label=self.classify_label)
        test_dataset = ClassifySubDataset(data_name=self.dataset, data_type='test',
                                          vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                          merge=self.merge, min_freq=self.min_freq,
                                          max_train_size=None,
                                          classify_label=self.classify_label)
        return train_dataset, eval_dataset, test_dataset


class ClassifySubDataset(TrajectoryProcessingDataset):
    def __init__(self, data_name, data_type, vocab, seq_len=512, add_cls=True,
                 merge=True, min_freq=1, max_train_size=None, classify_label='vflag'):
        self.vocab = vocab
        self.seq_len = seq_len
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self.classify_label = classify_label
        self._logger = getLogger()

        self.data_path = 'raw_data/{}/{}_{}.csv'.format(data_name, data_name, data_type)
        self.cache_path = 'raw_data/{}/cache_{}_{}_{}_{}_{}_{}.pkl'.format(
            data_name, data_name, data_type, add_cls, merge, min_freq, classify_label)
        self.temporal_mat_path = self.cache_path[:-4] + '_temporal_mat.pkl'
        self._load_data()

    def data_processing(self, origin_data):
        self._logger.info('Processing dataset in ClassifySubDataset!')
        sub_data = origin_data[['path', 'tlist', 'usr_id', 'traj_id', 'vflag']]
        traj_list = []
        temporal_mat_list = []
        for i in tqdm(range(sub_data.shape[0])):
            traj = sub_data.iloc[i]
            loc_list = eval(traj['path'])
            tim_list = eval(traj['tlist'])
            usr_id = traj['usr_id']
            vflag = int(traj['vflag'])
            # assert vflag == 0 or vflag == 1
            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.utcfromtimestamp(tim) for tim in tim_list]
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]  # 留一个PAD的位置
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]  # 留一个PAD的位置
            usr_list = [self.vocab.usr2index.get(usr_id, self.vocab.unk_index)] * len(new_loc_list)
            vflag_list = [vflag] * len(new_loc_list)
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list  # CLS的时间不重要
                vflag_list = [vflag_list[0]] + vflag_list
            temporal_mat = self._cal_mat(tim_list)  # (seq_len, seq_len)
            temporal_mat_list.append(temporal_mat)
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks, usr_list, vflag_list]).transpose((1, 0))  # (seq_length, feat_dim)
            traj_list.append(traj_fea)
        pickle.dump(traj_list, open(self.cache_path, 'wb'))
        pickle.dump(temporal_mat_list, open(self.temporal_mat_path, 'wb'))
        return traj_list, temporal_mat_list  # [loc, tim, mins, weeks, usr]


def collate_superv_classify_vflag(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)

    labels = []
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        labels.append(features[i][-1][5])
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    targets = torch.LongTensor(labels)  # (batch_size,)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    return X.long(), targets.long(), padding_masks, batch_temporal_mat.long()


def collate_superv_classify_usrid(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    labels = []
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        labels.append(features[i][-1][4])
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
    X[:, :, 4] = vocab.pad_index

    targets = torch.LongTensor(labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    return X.long(), targets.long(), padding_masks, batch_temporal_mat.long()
