import os
import json
import datetime
from logging import getLogger
import numpy as np
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.evaluator.utils import top_k
from libcity.utils import ensure_dir
from sklearn.metrics import f1_score


class MultiClassificationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self.metrics = config.get('metrics', ["Precision", "Recall", "F1", "MRR", "MAP", "NDCG",
                                              "microF1", "macroF1"])
        self.config = config
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self.topk = config.get('topk', [1])
        self.allowed_metrics = ['Precision', 'Recall', 'F1', 'MRR', 'MAP', 'NDCG',
                                "microF1", "macroF1"]
        self.clear()
        self._logger = getLogger()
        self._check_config()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in self.allowed_metrics:
                raise ValueError('the metric is not allowed in MultiClassificationEvaluator')

    def collect(self, batch):
        self.num_class = batch['loc_pred'].shape[-1]
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        total = len(batch['loc_true'])
        self.intermediate_result['total'] += total
        for k in self.topk:
            hit, rank, dcg = top_k(batch['loc_pred'], batch['loc_true'], k)
            self.intermediate_result['hit@' + str(k)] += hit
            self.intermediate_result['rank@' + str(k)] += rank
            self.intermediate_result['dcg@' + str(k)] += dcg

        pred = list(np.argmax(batch['loc_pred'], axis=1))
        truth = list(batch['loc_true'])
        assert len(pred) == len(truth)
        self.intermediate_result['pred'] += pred
        self.intermediate_result['truth'] += truth

    def evaluate(self):
        assert len(self.intermediate_result['pred']) == len(self.intermediate_result['truth'])
        self.result['microF1'] = f1_score(self.intermediate_result['truth'], self.intermediate_result['pred'],
                                          average='micro', labels=np.arange(self.num_class).tolist())
        self.result['macroF1'] = f1_score(self.intermediate_result['truth'], self.intermediate_result['pred'],
                                          average='macro', labels=np.arange(self.num_class).tolist())

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

        return self.result

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
        self.intermediate_result['truth'] = []
        self.intermediate_result['pred'] = []
