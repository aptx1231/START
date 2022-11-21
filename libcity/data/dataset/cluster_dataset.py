import json
import numpy as np
from logging import getLogger
from libcity.data.dataset import BaseDataset, TrajectoryProcessingDataset, padding_mask
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch
import datetime
import pickle
import pandas as pd


class ClusterDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.cluster_data_path = config.get('cluster_data_path', None)
        self._load_geo_latlon()
        self.collate_fn = collate_unsuperv_down

    def _load_geo_latlon(self):
        if self.dataset in ['bj', 'geolife']:
            self.geo_file = pd.read_csv('raw_data/bj_roadmap_edge/bj_roadmap_edge.geo')
        if self.dataset in ['porto']:
            self.geo_file = pd.read_csv('raw_data/porto_roadmap_edge/porto_roadmap_edge.geo')
        assert self.geo_file['type'][0] == 'LineString'
        self.geoid2latlon = {}
        for i in range(self.geo_file.shape[0]):
            geo_id = int(self.geo_file.iloc[i]['geo_id'])
            coordinates = eval(self.geo_file.iloc[i]['coordinates'])
            self.geoid2latlon[geo_id] = coordinates
        self._logger.info("Loaded Geo2LatLon, num_nodes=" + str(len(self.geoid2latlon)))

    def _gen_dataset(self):
        test_dataset = DownStreamSubDataset(data_name=self.dataset,
                                            data_path=self.cluster_data_path,
                                            vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                            max_train_size=None,
                                            geo2latlon=self.geoid2latlon)
        return [None], [None], test_dataset

    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls))
        return [None], [None], test_dataloader


class DownStreamSubDataset(TrajectoryProcessingDataset):
    def __init__(self, data_name, data_path, vocab, seq_len=512,
                 add_cls=True, max_train_size=None, geo2latlon=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self.geo2latlon = geo2latlon
        self._logger = getLogger()

        base_path = 'raw_data/{}/'.format(data_name)
        self.data_path = base_path + data_path + '.csv'
        self.cache_path = base_path + data_path + '_add_id.pkl'
        self.cache_path_wkt = base_path + data_path + '_add_id.json'
        self.temporal_mat_path = self.cache_path[:-4] + '_temporal_mat.pkl'
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.cache_path) and os.path.exists(self.temporal_mat_path) \
                and os.path.exists(self.cache_path_wkt):
            self.traj_list = pickle.load(open(self.cache_path, 'rb'))
            self.traj_wkt = json.load(open(self.cache_path_wkt, 'r'))
            self.temporal_mat_list = pickle.load(open(self.temporal_mat_path, 'rb'))
            self._logger.info('Load dataset from {}'.format(self.cache_path))
        else:
            origin_data_df = pd.read_csv(self.data_path, sep=';')
            self.traj_list, self.temporal_mat_list, self.traj_wkt = self.data_processing(origin_data_df)
        if self.max_train_size is not None:
            self.traj_list = self.traj_list[:self.max_train_size]
            self.temporal_mat_list = self.temporal_mat_list[:self.max_train_size]

    def data_processing(self, origin_data):
        self._logger.info('Processing dataset in DownStreamSubDataset!')
        sub_data = origin_data[['id', 'path', 'tlist', 'usr_id', 'traj_id', 'vflag']]
        traj_list = []
        traj_wkt = {}
        temporal_mat_list = []
        for i in tqdm(range(sub_data.shape[0]), desc=self.data_path):
            traj = sub_data.iloc[i]
            loc_list = eval(traj['path'])
            tim_list = eval(traj['tlist'])
            usr_id = traj['usr_id']
            vflag = int(traj['vflag'])
            # assert vflag == 0 or vflag == 1
            id_ = int(traj['id'])

            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.utcfromtimestamp(tim) for tim in tim_list]
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.usr2index.get(usr_id, self.vocab.unk_index)] * len(new_loc_list)
            vflag_list = [vflag] * len(new_loc_list)
            id_list = [id_] * len(new_loc_list)

            # cal wkt str
            wkt_str = 'LINESTRING('
            for j in range(len(loc_list)):
                rid = loc_list[j]
                coordinates = self.geo2latlon[rid]  # [(lat1, lon1), (lat2, lon2), ...]
                for coor in coordinates:
                    wkt_str += (str(coor[0]) + ' ' + str(coor[1]) + ',')
            if wkt_str[-1] == ',':
                wkt_str = wkt_str[:-1]
            wkt_str += ')'
            traj_wkt[id_] = wkt_str

            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list
                vflag_list = [vflag_list[0]] + vflag_list
                id_list = [id_list[0]] + id_list
            temporal_mat = self._cal_mat(tim_list)  # (seq_len, seq_len)
            temporal_mat_list.append(temporal_mat)
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks,
                                 usr_list, vflag_list, id_list]).transpose((1, 0))  # (seq_length, feat_dim)
            traj_list.append(traj_fea)
        pickle.dump(traj_list, open(self.cache_path, 'wb'))
        json.dump(traj_wkt, open(self.cache_path_wkt, 'w'))
        pickle.dump(temporal_mat_list, open(self.temporal_mat_path, 'wb'))
        return traj_list, temporal_mat_list, traj_wkt  # [loc, tim, mins, weeks, usr, vflag, id]


def collate_unsuperv_down(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    return X.long(), padding_masks, batch_temporal_mat.long()
