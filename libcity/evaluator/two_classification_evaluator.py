import os
import json
import datetime
from logging import getLogger

import numpy as np
import pandas as pd
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.utils import ensure_dir
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, cohen_kappa_score


class TwoClassificationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self.metrics = config.get('metrics', ["Accuracy", "Precision", "Recall", "F1", "AUC", "kappa"])
        self.config = config
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self.allowed_metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "kappa"]
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
        pred = list(np.argmax(batch['loc_pred'], axis=1))
        truth = list(batch['loc_true'])
        positive_prob = list(np.exp(batch['loc_pred'][:, 1]))  # 正类的概率
        assert len(pred) == len(truth)
        self.intermediate_result['pred'] += pred
        self.intermediate_result['truth'] += truth
        self.intermediate_result['positive_prob'] += positive_prob

    def evaluate(self):
        assert len(self.intermediate_result['pred']) == len(self.intermediate_result['truth'])
        self.result['Accuracy'] = accuracy_score(self.intermediate_result['truth'], self.intermediate_result['pred'])
        self.result['Recall'] = recall_score(self.intermediate_result['truth'], self.intermediate_result['pred'])
        self.result['Precision'] = precision_score(self.intermediate_result['truth'], self.intermediate_result['pred'])
        self.result['F1'] = f1_score(self.intermediate_result['truth'], self.intermediate_result['pred'])
        self.result['AUC'] = roc_auc_score(self.intermediate_result['truth'], self.intermediate_result['positive_prob'])
        self.result['kappa'] = cohen_kappa_score(self.intermediate_result['truth'], self.intermediate_result['pred'])
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
            dataframe = pd.DataFrame(self.result, index=[0])
            path = os.path.join(save_path, '{}.csv'.format(filename))
            dataframe.to_csv(path, index=False)
            self._logger.info('Evaluate result is saved at ' + path)
            self._logger.info("\n" + str(dataframe))
        return dataframe

    def clear(self):
        self.result = {}
        self.intermediate_result = dict()
        self.intermediate_result['truth'] = []
        self.intermediate_result['positive_prob'] = []
        self.intermediate_result['pred'] = []
