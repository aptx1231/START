import os
import torch
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libcity.utils import ensure_dir
from libcity.executor.scheduler import CosineLRScheduler
from libcity.utils import get_evaluator


class AbstractExecutor(object):

    def __init__(self, config, model, data_feature):
        self.config = config
        self.data_feature = data_feature
        self._logger = getLogger()

        self.vocab_size = self.data_feature.get('vocab_size')
        self.usr_num = self.data_feature.get('usr_num')

        self.exp_id = self.config.get('exp_id', None)
        self.device = self.config.get('device', torch.device('cpu'))
        self.epochs = self.config.get('max_epoch', 100)
        self.model_name = self.config.get('model', '')

        self.learner = self.config.get('learner', 'adamw')
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.grad_accmu_steps = self.config.get('grad_accmu_steps', 1)
        self.test_every = self.config.get('test_every', 10)

        self.lr_decay = self.config.get('lr_decay', True)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'cosinelr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.lr_warmup_epoch = self.config.get("lr_warmup_epoch", 5)
        self.lr_warmup_init = self.config.get("lr_warmup_init", 1e-6)
        self.t_in_epochs = self.config.get("t_in_epochs", True)

        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.log_batch = self.config.get('log_batch', 10)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.l2_reg = self.config.get('l2_reg', None)

        self.adj_mx = self.data_feature.get('adj_mx')
        self.node_features = self.data_feature.get('node_features')
        self.edge_index = self.data_feature.get('edge_index')
        self.loc_trans_prob = self.data_feature.get('loc_trans_prob')
        self.add_lap = self.config.get('add_lap', True)
        self.graph_dict = {
            'node_features': self.node_features,
            'edge_index': self.edge_index,
            'loc_trans_prob': self.loc_trans_prob,
        }

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.png_dir = './libcity/cache/{}'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './libcity/cache/{}'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.png_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)
        self._writer = SummaryWriter(self.summary_writer_dir)

        self.model = model.to(self.device)  # bertlm
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self.optimizer.zero_grad()

        self.evaluator = get_evaluator(self.config, self.data_feature)

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model'] = self.model.cpu()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(config, cache_name)
        self.model.to(self.device)
        self._logger.info("Saved model at " + cache_name)

    def load_model_state(self, cache_name):
        """
        加载对应模型的 cache （用于加载参数直接进行测试的场景）

        Args:
            cache_name(str): 保存的文件名
        """
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'].state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at " + cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at " + cache_name)

    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model'] = self.model.cpu()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self.model.to(self.device)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            elif self.lr_scheduler_type.lower() == 'cosinelr':
                lr_scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=self.epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                    warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init, t_in_epochs=self.t_in_epochs)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor train not implemented")

    def _train_epoch(self, train_dataloader, epoch_idx):
        raise NotImplementedError("Executor evaluate not implemented")

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        raise NotImplementedError("Executor evaluate not implemented")

    def _draw_png(self, data):
        raise NotImplementedError("Executor evaluate not implemented")

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor evaluate not implemented")
