import os
import torch
from libcity.executor.nextloc_executor import NextLocExecutor
from libcity.model.trajectory_embedding import LinearClassify


class TrajClassifyExecutor(NextLocExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.criterion = torch.nn.NLLLoss(reduction='none')

    def _load_pretrain_model(self):
        checkpoint = torch.load(self.pretrain_model_path, map_location=torch.device('cpu'))
        pretrained_model = checkpoint['model']
        self._logger.info('Load Pretrained-Model from {}'.format(self.pretrain_model_path))
        if isinstance(self.model, LinearClassify):
            self.model.model.bert.load_state_dict(pretrained_model.bert.state_dict())
        else:
            raise ValueError('No bert in model!')
        if self.freeze:
            self._logger.info('Freeze the bert model parameters!')
            for pa in self.model.bert.parameters():
                pa.requires_grad = False

    def load_model_with_initial_ckpt(self, initial_ckpt):
        assert os.path.exists(initial_ckpt), 'Weights at %s not found' % initial_ckpt
        checkpoint = torch.load(initial_ckpt, map_location='cpu')
        pretrained_model = checkpoint['model'].bert.state_dict()
        self._logger.info('Load Pretrained-Model from {}'.format(initial_ckpt))

        if isinstance(self.model, LinearClassify):
            model_keys = self.model.model.bert.state_dict()
        else:
            raise ValueError('No bert in model!')

        state_dict_load = {}
        unexpect_keys = []
        for k, v in pretrained_model.items():  # 预训练模型
            if k not in model_keys.keys() or v.shape != model_keys[k].shape\
                    or self._valid_parameter(k):
                unexpect_keys.append(k)
            else:
                state_dict_load[k] = v
        for k, v in model_keys.items():  # 当前模型
            if k not in pretrained_model.keys():
                unexpect_keys.append(k)
        self._logger.info("Unexpected keys: {}".format(unexpect_keys))

        if isinstance(self.model, LinearClassify):
            self.model.model.bert.load_state_dict(state_dict_load, strict=False)
        else:
            raise ValueError('No bert in model!')
        self._logger.info("Initialize model from {}".format(initial_ckpt))
