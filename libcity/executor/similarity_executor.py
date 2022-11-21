import torch
import time
from libcity.executor.bert_executor import BertBaseExecutor
from libcity.model.trajectory_embedding import LinearSim
from tqdm import tqdm
import numpy as np


class SimilarityExecutor(BertBaseExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.pretrain_model_path = self.config.get("pretrain_path", None)
        self.freeze = self.config.get("freeze", False)
        if self.pretrain_model_path is not None:
            self._load_pretrain_model()

    def _load_pretrain_model(self):
        checkpoint = torch.load(self.pretrain_model_path, map_location=torch.device('cpu'))
        pretrained_model = checkpoint['model']
        self._logger.info('Load Pretrained-Model from {}'.format(self.pretrain_model_path))
        if isinstance(self.model, LinearSim):
            self.model.model.bert.load_state_dict(pretrained_model.bert.state_dict())
        else:
            raise ValueError('No bert in model!')
        if self.freeze:
            self._logger.info('Freeze the bert model parameters!')
            for pa in self.model.bert.parameters():
                pa.requires_grad = False

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        self.evaluator.save_result(self.evaluate_res_dir)

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        # database_dataloader, detour_dataloader, query_dataloader
        database_pred_list, database_id_list = self._do_prediction(train_dataloader, 'database')
        detour_pred_list, detour_id_list = self._do_prediction(eval_dataloader, 'detour')
        query_pred_list, query_id_list = self._do_prediction(test_dataloader, 'query')
        self.evaluator.clear()
        self.evaluator.collect([database_pred_list, database_id_list, detour_pred_list,
                                detour_id_list, query_pred_list, query_id_list])

    def _do_prediction(self, dataloader, desc):
        self.model = self.model.eval()
        pred_list = []
        id_list = []
        test_time_list = []
        with torch.no_grad():
            start_time = time.time()
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=desc):
                X, padding_masks, batch_temporal_mat = batch
                # X: (batch_size, padded_length, feat_dim)
                # padding_masks: (batch_size, padded_length)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X = X.to(self.device)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                batch_temporal_mat = batch_temporal_mat.to(self.device)
                start = time.time()
                predictions = self.model(
                    x=X, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                    graph_dict=self.graph_dict)  # (batch_size, feat_dim)
                end = time.time()
                test_time_list.append(end - start)
                pred_list.append(predictions.cpu().numpy())
                id_list.append(X.cpu().numpy()[..., -1][:, 0])
            end_time = time.time()
        self._logger.info(desc + ', total time = {} / {}'.format(sum(test_time_list), end_time - start_time))
        pred_list = np.concatenate(pred_list)  # (n, dim)
        id_list = np.concatenate(id_list)  # (n, )
        self._logger.info(desc + ', pred.shape={}, ids.shape={}'.format(pred_list.shape, id_list.shape))
        return pred_list, id_list

    def _train_epoch(self, train_dataloader, epoch_idx):
        pass

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        pass
