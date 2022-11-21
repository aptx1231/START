import os
import json
import time
import random
import numpy as np
import pandas as pd
from logging import getLogger

import torch
from sklearn.metrics import euclidean_distances
from libcity.evaluator.abstract_evaluator import AbstractEvaluator


class SimilarityEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self.metrics = config.get('metrics', ["MR", "MRR", "HR", "Precision"])
        self.config = config
        self.topk = config.get('topk', [1])
        self.sim_mode = config.get('sim_mode', 'most')  # most or knn
        self.allowed_metrics = ["MR", "MRR", "HR", "Precision"]
        self.clear()
        self._logger = getLogger()
        self._check_config()

        self.result = {}
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.d_model = config.get('d_model', 768)
        self.sim_select_num = config.get('sim_select_num', 5)

        self.query_data_path = config.get('query_data_path', None)
        self.detour_data_path = config.get('detour_data_path', None)
        self.origin_big_data_path = config.get('origin_big_data_path', None)
        base_path = 'raw_data/{}/'.format(self.dataset)
        self.query_wkt = json.load(open(base_path + self.query_data_path + '_add_id.json', 'r'))
        self.detour_wkt = json.load(open(base_path + self.detour_data_path + '_add_id.json', 'r'))
        self.database_wkt = json.load(open(base_path + self.origin_big_data_path + '_add_id.json', 'r'))

        self._init_path()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in self.allowed_metrics:
                raise ValueError('the metric is not allowed in ClassificationEvaluator')

    def _init_path(self):
        self.query_ids_path = './libcity/cache/{}/evaluate_cache/{}_query_ids_{}_{}_{}.npy'\
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.query_vec_path = './libcity/cache/{}/evaluate_cache/{}_query_vec_{}_{}_{}.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.detour_ids_path = './libcity/cache/{}/evaluate_cache/{}_detour_ids_{}_{}_{}.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.detour_vec_path = './libcity/cache/{}/evaluate_cache/{}_detour_vec_{}_{}_{}.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.database_ids_path = './libcity/cache/{}/evaluate_cache/{}_database_ids_{}_{}_{}.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.database_vec_path = './libcity/cache/{}/evaluate_cache/{}_database_vec_{}_{}_{}.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.euclidean_path = './libcity/cache/{}/evaluate_cache/{}_euclidean_{}_{}_{}_most.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.euclidean_path_truth = './libcity/cache/{}/evaluate_cache/{}_euclidean_truth_{}_{}_{}_knn.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.euclidean_path_pred = './libcity/cache/{}/evaluate_cache/{}_euclidean_pred_{}_{}_{}_knn.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.euclidean_index_path = './libcity/cache/{}/evaluate_cache/{}_euclidean_index_{}_{}_{}_most.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.evaluate_res_path = './libcity/cache/{}/evaluate_cache/{}_evaluate_res_{}_{}_{}_{}.json' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model, self.sim_mode)
        self.qgis_res_path = './libcity/cache/{}/evaluate_cache/{}_qgis_res_{}_{}_{}_{}.csv' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model, self.sim_mode)
        self.knn_hit_path = './libcity/cache/{}/evaluate_cache/{}_knn_hit_{}_{}_{}.npy' \
            .format(self.exp_id, self.exp_id, self.model, self.dataset, self.d_model)
        self.database_pred_list, self.database_id_list, self.detour_pred_list, \
            self.detour_id_list, self.query_pred_list, self.query_id_list = None, None, None, None, None, None

    def collect(self, batch):
        self.database_pred_list, self.database_id_list, self.detour_pred_list, \
            self.detour_id_list, self.query_pred_list, self.query_id_list = batch
        self._logger.info('Total query trajectory number = {}'.format(len(self.query_id_list)))
        self._logger.info('Total database trajectory number = {}'.format(
            len(self.database_id_list) + len(self.detour_id_list)))
        np.save(self.database_vec_path, self.database_pred_list)  # len=b
        np.save(self.database_ids_path, self.database_id_list)
        np.save(self.detour_vec_path, self.detour_pred_list)  # len=a
        np.save(self.detour_ids_path, self.detour_id_list)
        np.save(self.query_vec_path, self.query_pred_list)  # len=a
        np.save(self.query_ids_path, self.query_id_list)

    def evaluate(self):
        if self.database_pred_list is None:
            self.database_pred_list = np.load(self.database_vec_path)
            self.database_id_list = np.load(self.database_ids_path)
            self.detour_pred_list = np.load(self.detour_vec_path)
            self.detour_id_list = np.load(self.detour_ids_path)
            self.query_pred_list = np.load(self.query_vec_path)
            self.query_id_list = np.load(self.query_ids_path)
        if self.sim_mode == 'most':
            return self.evaluate_most_sim()
        elif self.sim_mode == 'knn':
            return self.evaluate_knn_sim()
        else:
            raise ValueError('Error evaluator similarity mode {}'.format(self.sim_mode))

    def evaluate_knn_sim(self):
        assert len(self.topk) == 1
        self.topk = self.topk[0]  # list to int
        t1, t2, t3 = 0, 0, 0
        if os.path.exists(self.euclidean_path_truth) and os.path.exists(self.euclidean_path_pred):
            eul_res_query = np.load(self.euclidean_path_truth)
            eul_res_detour = np.load(self.euclidean_path_pred)
        else:
            start_time = time.time()
            eul_res_query = euclidean_distances(self.query_pred_list, self.database_pred_list)  # (a, b)
            t1 = time.time() - start_time
            self._logger.info('Euclidean_distances Truth cost time {}.'.format(t1))
            np.save(self.euclidean_path_truth, eul_res_query)
            self._logger.info('Euclidean_distances Truth is saved at {}, shape={}.'.format(
                self.euclidean_path_truth, eul_res_query.shape))

            start_time = time.time()
            eul_res_detour = euclidean_distances(self.detour_pred_list, self.database_pred_list)  # (a, b)
            t1 = time.time() - start_time
            self._logger.info('Euclidean_distances Pred cost time {}.'.format(t1))
            np.save(self.euclidean_path_pred, eul_res_detour)
            self._logger.info('Euclidean_distances Pred is saved at {}, shape={}.'.format(
                self.euclidean_path_pred, eul_res_detour.shape))

        eul_res_query = torch.from_numpy(eul_res_query)
        start_time = time.time()
        _, eul_res_query_index = torch.topk(eul_res_query, self.topk, dim=1, largest=False)
        t2 = time.time() - start_time
        self._logger.info('Sorted euclidean_index Truth cost time {}.'.format(t2))
        eul_res_query_index = eul_res_query_index.cpu().numpy()

        eul_res_detour = torch.from_numpy(eul_res_detour)
        start_time = time.time()
        _, eul_res_detour_index = torch.topk(eul_res_detour, self.topk, dim=1, largest=False)
        t2 = time.time() - start_time
        self._logger.info('Sorted euclidean_index Pred cost time {}.'.format(t2))
        eul_res_detour_index = eul_res_detour_index.cpu().numpy()

        start_time = time.time()
        total_num = eul_res_query_index.shape[0]
        hit = []
        for i in range(total_num):
            query_k = set(eul_res_query_index[i].tolist())
            detour_k = eul_res_detour_index[i].tolist()
            cnt = 0
            for ind in detour_k:
                if ind in query_k:
                    cnt += 1
            hit.append(cnt)
        np.save(self.knn_hit_path, np.array(hit))
        self.result['Precision'] = (1.0 * sum(hit)) / (total_num * self.topk)
        t3 = time.time() - start_time
        self._logger.info("Evaluate cost time is {}".format(t3))
        self._logger.info("Evaluate result is {}".format(self.result))
        json.dump(self.result, open(self.evaluate_res_path, 'w'), indent=4)
        self._logger.info('Evaluate result is saved at {}'.format(self.evaluate_res_path))
        self._logger.info("Total cost time is {}".format(t1 + t2 + t3))

        select_index = np.arange(total_num)
        random.shuffle(select_index)
        select_index = select_index[:self.sim_select_num]
        output = []
        for i in select_index:
            # query
            output.append([str(i) + '-query', self.query_id_list[i], self.query_wkt[str(self.query_id_list[i])], i])
            # detour
            output.append([str(i) + '-detour', self.detour_id_list[i], self.detour_wkt[str(self.detour_id_list[i])], i])
            # query topk-sim
            for ind in eul_res_query_index[i].tolist():
                output.append([str(i) + '-query-' + str(ind), self.database_id_list[ind],
                               self.database_wkt[str(self.database_id_list[ind])], i])
            # detour topk-sim
            for ind in eul_res_detour_index[i].tolist():
                output.append([str(i) + '-detour-' + str(ind), self.database_id_list[ind],
                               self.database_wkt[str(self.database_id_list[ind])], i])
        output = pd.DataFrame(output, columns=['index', 'id', 'wkt', 'class'])
        output.to_csv(self.qgis_res_path, index=False)
        return self.result

    def evaluate_most_sim(self):
        t1, t2, t3 = 0, 0, 0
        if os.path.exists(self.euclidean_path):
            eul_res = np.load(self.euclidean_path)
        else:
            start_time = time.time()
            database_all = np.concatenate([self.detour_pred_list, self.database_pred_list], axis=0)
            eul_res = euclidean_distances(self.query_pred_list, database_all)  # (a, a+b)
            t1 = time.time() - start_time
            self._logger.info('Euclidean_distances cost time {}.'.format(t1))
            np.save(self.euclidean_path, eul_res)
            self._logger.info('Euclidean_distances is saved at {}, shape={}.'.format(
                self.euclidean_path, eul_res.shape))

        if os.path.exists(self.euclidean_index_path):
            sorted_eul_index = np.load(self.euclidean_index_path)
        else:
            start_time = time.time()
            sorted_eul_index = eul_res.argsort(axis=1)
            t2 = time.time() - start_time
            self._logger.info('Sorted euclidean_index cost time {}.'.format(t2))
            np.save(self.euclidean_index_path, sorted_eul_index)
            self._logger.info('Sorted euclidean_index is saved at {}, shape={}.'.format(
                self.euclidean_index_path, sorted_eul_index.shape))

        start_time = time.time()
        total_num = eul_res.shape[0]
        hit = {}
        for k in self.topk:
            hit[k] = 0
        rank = 0
        rank_p = 0.0
        for i in range(total_num):
            rank_list = list(sorted_eul_index[i])
            rank_index = rank_list.index(i)
            # rank_index is start from 0, so need plus 1
            rank += (rank_index + 1)
            rank_p += 1.0 / (rank_index + 1)
            for k in self.topk:
                if i in sorted_eul_index[i][:k]:
                    hit[k] += 1

        self.result['MR'] = rank / total_num
        self.result['MRR'] = rank_p / total_num
        for k in self.topk:
            self.result['HR@{}'.format(k)] = hit[k] / total_num
        t3 = time.time() - start_time
        self._logger.info("Evaluate cost time is {}".format(t3))
        self._logger.info("Evaluate result is {}".format(self.result))
        json.dump(self.result, open(self.evaluate_res_path, 'w'), indent=4)
        self._logger.info('Evaluate result is saved at {}'.format(self.evaluate_res_path))
        self._logger.info("Total cost time is {}".format(t1 + t2 + t3))

        kmax = max(self.topk)
        select_index = np.arange(len(sorted_eul_index))
        random.shuffle(select_index)
        select_index = select_index[:self.sim_select_num]
        output = []
        for i in select_index:
            # query
            output.append([str(i) + '-query', self.query_id_list[i], self.query_wkt[str(self.query_id_list[i])], i])
            # detour
            output.append([str(i) + '-detour', self.detour_id_list[i], self.detour_wkt[str(self.detour_id_list[i])], i])
            # tok-sim
            for ind, d in enumerate(sorted_eul_index[i, 0:kmax]):
                index_out = str(i) + '-' + str(ind)
                if d >= len(self.detour_id_list):  # 大数据库中的轨迹
                    d -= len(self.detour_id_list)
                    output.append([index_out, self.database_id_list[d],
                                   self.database_wkt[str(self.database_id_list[d])], i])
                else:
                    if d == i:
                        index_out += '-find'
                    output.append([index_out, self.detour_id_list[d],
                                   self.detour_wkt[str(self.detour_id_list[d])], i])
        output = pd.DataFrame(output, columns=['index', 'id', 'wkt', 'class'])
        output.to_csv(self.qgis_res_path, index=False)
        return self.result

    def save_result(self, save_path, filename=None):
        self.evaluate()

    def clear(self):
        self.result = {}
