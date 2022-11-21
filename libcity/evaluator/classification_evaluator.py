import os
import json
import datetime
from logging import getLogger
import pandas as pd
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.evaluator.utils import top_k
from libcity.utils import ensure_dir


class ClassificationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self.metrics = config.get('metrics', ["Precision", "Recall", "F1", "MRR", "MAP", "NDCG"])
        self.config = config
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self.topk = config.get('topk', [1])
        self.allowed_metrics = ['Precision', 'Recall', 'F1', 'MRR', 'MAP', 'NDCG']
        self.clear()
        self._logger = getLogger()
        self._check_config()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in self.allowed_metrics:
                raise ValueError('the metric is not allowed in ClassificationEvaluator')

    def collect(self, batch):
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        total = len(batch['loc_true'])
        self.intermediate_result['total'] += total
        for k in self.topk:
            hit, rank, dcg = top_k(batch['loc_pred'], batch['loc_true'], k)
            self.intermediate_result['hit@' + str(k)] += hit
            self.intermediate_result['rank@' + str(k)] += rank
            self.intermediate_result['dcg@' + str(k)] += dcg

    def evaluate(self):
        for k in self.topk:
            precision = self.intermediate_result['hit@{}'.format(k)] / (self.intermediate_result['total'] * k)
            self.result['Precision@{}'.format(k)] = precision

            recall = self.intermediate_result['hit@{}'.format(k)] / self.intermediate_result['total']
            self.result['Recall@{}'.format(k)] = recall

            if precision + recall == 0:
                self.result['F1@{}'.format(k)] = 0.0
            else:
                self.result['F1@{}'.format(k)] = (2 * precision * recall) / (precision + recall)

            self.result['MRR@{}'.format(k)] = \
                self.intermediate_result['rank@{}'.format(k)] / self.intermediate_result['total']

            self.result['MAP@{}'.format(k)] = \
                self.intermediate_result['rank@{}'.format(k)] / self.intermediate_result['total']

            self.result['NDCG@{}'.format(k)] = \
                self.intermediate_result['dcg@{}'.format(k)] / self.intermediate_result['total']
        return self.result

    def save_result(self, save_path, filename=None):
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:
            filename = str(self.config['exp_id']) + '_' + \
                       datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is {}'.format(json.dumps(self.result, indent=1)))
            path = os.path.join(save_path, '{}.json'.format(filename))
            with open(path, 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at {}'.format(path))

        dataframe = {}
        if 'csv' in self.save_modes:
            for metric in self.metrics:
                dataframe[metric] = []
            for metric in self.metrics:
                for k in self.topk:
                    dataframe[metric].append(self.result[metric + '@' + str(k)])
            dataframe = pd.DataFrame(dataframe, index=self.topk)
            path = os.path.join(save_path, '{}.csv'.format(filename))
            dataframe.to_csv(path, index=False)
            self._logger.info('Evaluate result is saved at ' + path)
            self._logger.info("\n" + str(dataframe))
        return dataframe

    def clear(self):
        self.result = {}
        self.intermediate_result = dict()
        self.intermediate_result['total'] = 0
        for inter in ['hit']:
            for k in self.topk:
                self.intermediate_result[inter + '@' + str(k)] = 0
        for inter in ['rank', 'dcg']:
            for k in self.topk:
                self.intermediate_result[inter + '@' + str(k)] = 0.0
