import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from libcity.data.dataset import WordVocab
from libcity.utils import str2bool, ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument('--roadnetwork', type=str, default='porto_roadmap_edge', help='road network dataset')
parser.add_argument('--dataset', type=str, default='porto', help='the name of dataset')
parser.add_argument('--min_freq', type=int, default=1, help='Minimum frequency of occurrence of road segments')
parser.add_argument('--use_mask', type=str2bool, default=True, help='Whether to use mask or not in vocab')
parser.add_argument('--merge', type=str2bool, default=True, help='Whether to merge 3 dataset to get vocab')
parser.add_argument('--seq_len', type=int, default=128, help='max len of trajectory')
parser.add_argument('--bidir_adj_mx', type=str2bool, default=False, help='whether bidir the adj_mx')
args = parser.parse_args()

data_name = args.dataset
use_mask = args.use_mask
min_freq = args.min_freq
seq_len = args.seq_len
roadnetwork = args.roadnetwork

roadmap_path = 'raw_data/{}/{}'.format(roadnetwork, roadnetwork)
traj_path_train = 'raw_data/{}/{}_train.csv'.format(data_name, data_name)
traj_path_val = 'raw_data/{}/{}_eval.csv'.format(data_name, data_name)
traj_path_test = 'raw_data/{}/{}_test.csv'.format(data_name, data_name)

if args.merge:
    traj_path_merge = 'raw_data/{}/{}_merge.csv'.format(data_name, data_name)
    if not os.path.exists(traj_path_merge):
        train = pd.read_csv(traj_path_train, sep=';')
        vals = pd.read_csv(traj_path_val, sep=';')
        test = pd.read_csv(traj_path_test, sep=';')
        merge = pd.concat([train, vals, test], axis=0)
        merge.to_csv(traj_path_merge, sep=';', index=False)
    vocab_path = 'raw_data/vocab_{}_{}_{}_merge.pkl'.format(data_name, use_mask, min_freq)
    traj_path = traj_path_merge
else:
    vocab_path = 'raw_data/vocab_{}_{}_{}.pkl'.format(data_name, use_mask, min_freq)
    traj_path = traj_path_train

if not os.path.exists(vocab_path):
    vocab = WordVocab(traj_path=traj_path, roadmap_path=roadmap_path,
                      min_freq=min_freq, use_mask=use_mask, seq_len=seq_len)
    vocab.save_vocab(vocab_path)
    print("VOCAB SIZE ", len(vocab))
else:
    vocab = WordVocab.load_vocab(vocab_path)
print('user num ', vocab.user_num)
print("vocab size ", vocab.vocab_size)
print("del edge ", vocab.del_edge)
print("len(vocab.all_edge) ", len(vocab.all_edge))
print(vocab.to_seq([16104, 15665, 18751, 40088, 21759, 18690, 40304]))
print(vocab.from_seq(vocab.to_seq([16104, 15665, 18751, 40088, 21759, 18690, 40304])))


def select_geo_rel(selected_geo_ids, roadnetwork, data_name, use_mask, min_freq, merge):
    new_data_name = '{}_{}_{}_{}'.format(roadnetwork, data_name, use_mask, min_freq)
    if merge:
        new_data_name += '_merge'
    selected_path = 'raw_data/{}'.format(new_data_name)
    ensure_dir(selected_path)
    selected_geo_ids = set(selected_geo_ids)

    if os.path.exists(selected_path + '/{}.geo'.format(new_data_name)) and \
            os.path.exists(selected_path + '/{}.rel'.format(new_data_name)):
        return

    geofile = pd.read_csv('raw_data/{}/{}'.format(roadnetwork, roadnetwork) + '.geo')
    geo = []
    for i in tqdm(range(geofile.shape[0]), desc='geo'):
        if int(geofile.iloc[i]['geo_id']) in selected_geo_ids:
           geo.append(geofile.iloc[i].values.tolist())
    geo = pd.DataFrame(geo, columns=geofile.columns)
    geo.to_csv(selected_path + '/{}.geo'.format(new_data_name), index=False)

    relfile = pd.read_csv('raw_data/{}/{}'.format(roadnetwork, roadnetwork) + '.rel')
    rel = []
    for i in tqdm(range(relfile.shape[0]), desc='rel'):
        oid = relfile.iloc[i]['origin_id']
        did = relfile.iloc[i]['destination_id']
        if oid not in selected_geo_ids or did not in selected_geo_ids:
            continue
        rel.append(relfile.iloc[i].values.tolist())
    rel = pd.DataFrame(rel, columns=relfile.columns)
    rel.to_csv(selected_path + '/{}.rel'.format(new_data_name), index=False)

    config = {"info": {
        "geo_file": new_data_name,
        "rel_file": new_data_name
    }}
    json.dump(config, open(selected_path + '/config.json', 'w'), indent=4)


select_geo_rel(vocab.all_edge, roadnetwork, data_name, use_mask, min_freq, args.merge)


def append_degree(roadnetwork, data_name, use_mask, min_freq, merge, bidir_adj_mx):
    new_data_name = '{}_{}_{}_{}'.format(roadnetwork, data_name, use_mask, min_freq)
    if merge:
        new_data_name += '_merge'
    selected_path = 'raw_data/{}'.format(new_data_name)
    ensure_dir(selected_path)

    if os.path.exists(selected_path + '/{}_withdegree.geo'.format(new_data_name)) and \
            os.path.exists(selected_path + '/{}_withdegree.rel'.format(new_data_name)):
        return

    geo_file = selected_path + '/{}.geo'.format(new_data_name)
    rel_file = selected_path + '/{}.rel'.format(new_data_name)

    geo = pd.read_csv(geo_file)
    rel = pd.read_csv(rel_file)[['origin_id', 'destination_id']]

    geo_ids = list(geo['geo_id'])
    geo_to_ind = {}
    ind_to_geo = {}
    for index, geo_id in enumerate(geo_ids):
        geo_to_ind[geo_id] = index
        ind_to_geo[index] = geo_id

    adj_mx = np.zeros((len(geo_ids), len(geo_ids)), dtype=np.float32)
    for row in rel.values:
        if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
            print(row[0], row[1])
            continue
        adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = 1
        if bidir_adj_mx:
            adj_mx[geo_to_ind[row[1]], geo_to_ind[row[0]]] = 1

    outdegree = np.sum(adj_mx, axis=1)  # (N, )
    indegree = np.sum(adj_mx.T, axis=1)  # (N, )
    outdegree_list = []
    indegree_list = []

    for i, row in tqdm(geo.iterrows(), total=geo.shape[0]):
        geo_id = row.geo_id
        outdegree_i = outdegree[geo_to_ind[geo_id]]
        indegree_i = indegree[geo_to_ind[geo_id]]
        outdegree_list.append(int(outdegree_i))
        indegree_list.append(int(indegree_i))

    geo.insert(loc=geo.shape[1], column='outdegree', value=outdegree_list)
    geo.insert(loc=geo.shape[1], column='indegree', value=indegree_list)

    rel.to_csv(rel_file[:-4] + '_withdegree.rel', index=False)
    geo.to_csv(geo_file[:-4] + '_withdegree.geo', index=False)


append_degree(roadnetwork, data_name, use_mask, min_freq, args.merge, args.bidir_adj_mx)
