import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from libcity.utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='porto', help='the name of dataset')
parser.add_argument('--K', type=int, default=1, help='K step')  # K步转移、K步可达
parser.add_argument('--seq_len', type=int, default=128, help='max len of trajectory')
parser.add_argument('--bidir_adj_mx', type=str2bool, default=False, help='whether bidir the adj_mx')  # 永远是False
parser.add_argument('--custom', type=bool, default=True, help='custom matrix matmul')
args = parser.parse_args()

base_path = 'raw_data'
data_name = args.dataset
K = args.K
bidir_adj_mx = args.bidir_adj_mx

if data_name == 'bj':
    road_name = 'bj_roadmap_edge_bj_True_1_merge'
elif data_name == 'porto':
    road_name = 'porto_roadmap_edge_porto_True_1_merge'

rel_file = os.path.join(base_path, '{0}/{0}.rel'.format(road_name))
geo_file = os.path.join(base_path, '{0}/{0}.geo'.format(road_name))

geo = pd.read_csv(geo_file)
print('Geo', geo.shape)

geo_ids = list(geo['geo_id'])
num_nodes = len(geo_ids)
geo_to_ind = {}
ind_to_geo = {}
for index, geo_id in enumerate(geo_ids):
    geo_to_ind[geo_id] = index
    ind_to_geo[index] = geo_id


def cal_matmul(mat1, mat2):
    n = mat1.shape[0]
    assert mat1.shape[0] == mat1.shape[1] == mat2.shape[0] == mat2.shape[1]
    res = np.zeros((n, n), dtype='bool')
    for i in tqdm(range(n), desc='outer'):
        for j in tqdm(range(n), desc='inner'):
            res[i, j] = np.dot(mat1[i, :], mat2[:, j])
    return res


path = os.path.join(base_path, '{0}/{0}_neighbors_{1}.json'.format(road_name, K))
if os.path.exists(path):
    geoid2neighbors = json.load(open(path, 'r'))
else:
    relfile = pd.read_csv(rel_file)[['origin_id', 'destination_id']]
    print('Rel', relfile.shape)
    adj_mx = np.zeros((len(geo_ids), len(geo_ids)), dtype=np.float32)
    for row in relfile.values:
        if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
            continue
        adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = 1
        if bidir_adj_mx:
            adj_mx[geo_to_ind[row[1]], geo_to_ind[row[0]]] = 1

    adj_mx_bool = adj_mx.astype('bool')
    k_adj_mx_list = [adj_mx_bool]
    for i in tqdm(range(2, K + 1)):
        if args.custom:
            k_adj_mx_list.append(cal_matmul(k_adj_mx_list[-1], adj_mx_bool))
        else:
            k_adj_mx_list.append(np.matmul(k_adj_mx_list[-1], adj_mx_bool))
        np.save(os.path.join(base_path, '{0}/{0}_adj_{1}.npy'.format(road_name, i)), k_adj_mx_list[-1])
    print('Finish K order adj_mx')
    for i in tqdm(range(1, len(k_adj_mx_list))):
        adj_mx_bool += k_adj_mx_list[i]
    print('Finish sum of K order adj_mx')
    geoid2neighbors = {}
    for i in tqdm(range(len(adj_mx_bool)), desc='count neighbors'):
        geo_id = int(ind_to_geo[i])
        geoid2neighbors[geo_id] = []
        for j in range(adj_mx_bool.shape[1]):
            if adj_mx_bool[i][j] == 0:
                continue
            ner_id = int(ind_to_geo[j])
            geoid2neighbors[geo_id].append(ner_id)
    json.dump(geoid2neighbors, open(path, 'w'))
    print('Total edge@{} = {}'.format(1, adj_mx.sum()))
    print('Total edge@{} = {}'.format(K, adj_mx_bool.sum()))

path = os.path.join(base_path, '{0}/{0}_trans_prob_{1}.json'.format(road_name, K))
if os.path.exists(path):
    link2prob = json.load(open(path, 'r'))
else:
    node_array = np.zeros([num_nodes, num_nodes], dtype=float)
    print(node_array.shape)
    count_array_row = np.zeros([num_nodes], dtype=int)
    count_array_col = np.zeros([num_nodes], dtype=int)

    train_file = 'raw_data/{}/{}_train.csv'.format(data_name, data_name)

    train = pd.read_csv(train_file, sep=';', dtype={'id': int, 'vflag': int, 'hop': int, 'traj_id': int})

    max_length = args.seq_len
    for _, row in tqdm(train.iterrows(), total=train.shape[0], desc='count traj prob'):
        plist = eval(row.path)[:max_length]
        for i in range(len(plist) - 1):
            prev_geo = plist[i]
            for j in range(1, K+1):
                if i + j >= len(plist):
                    continue
                next_geo = plist[i + j]
                prev_ind = geo_to_ind[prev_geo]
                next_ind = geo_to_ind[next_geo]
                count_array_row[prev_ind] += 1
                count_array_col[next_ind] += 1
                node_array[prev_ind][next_ind] += 1

    assert (count_array_row == (node_array.sum(1))).sum() == len(count_array_row)  # 按行求和
    assert (count_array_col == (node_array.sum(0))).sum() == len(count_array_col)  # 按列求和

    node_array_out = node_array.copy()
    for i in tqdm(range(node_array_out.shape[0])):
        count = count_array_row[i]
        if count == 0:
            print(i, 'no out-degree')
            continue
        node_array_out[i, :] /= count

    node_array_in = node_array.copy()
    for i in tqdm(range(node_array_in.shape[0])):
        count = count_array_col[i]
        if count == 0:
            print(i, 'no in-degree')
            continue
        node_array_in[:, i] /= count

    # rel_file = os.path.join(base_path, '{0}/{0}_withdegree.rel'.format(road_name))
    # rel = pd.read_csv(rel_file)
    # for i, row in tqdm(rel.iterrows(), total=rel.shape[0]):
    #     prev_id = row.origin_id
    #     next_id = row.destination_id
    #     rel.loc[i, 'outprob_{}'.format(K)] = node_array_out[geo_to_ind[prev_id]][geo_to_ind[next_id]]
    #     rel.loc[i, 'inprob_{}'.format(K)] = node_array_in[geo_to_ind[prev_id]][geo_to_ind[next_id]]
    # rel.to_csv(rel_file, index=False)

    link2prob = {}
    for k, v in geoid2neighbors.items():
        for tgt in v:
            id_ = str(k) + '_' + str(tgt)
            p = node_array_in[geo_to_ind[int(k)]][geo_to_ind[int(tgt)]]
            link2prob[id_] = float(p)
    path = os.path.join(base_path, '{0}/{0}_trans_prob_{1}.json'.format(road_name, K))
    json.dump(link2prob, open(path, 'w'))
