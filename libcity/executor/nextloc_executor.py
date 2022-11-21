import os
import torch
from libcity.executor.bert_executor import BertBaseExecutor
from libcity.model import loss
from libcity.model.trajectory_embedding import LinearNextLoc
from tqdm import tqdm


class NextLocExecutor(BertBaseExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.criterion = torch.nn.NLLLoss(reduction='none')
        self.pretrain_model_path = self.config.get("pretrain_path", None)
        self.freeze = self.config.get("freeze", False)
        if self.pretrain_model_path is not None:
            self._load_pretrain_model()

    def _load_pretrain_model(self):
        checkpoint = torch.load(self.pretrain_model_path, map_location=torch.device('cpu'))
        pretrained_model = checkpoint['model']
        self._logger.info('Load Pretrained-Model from {}'.format(self.pretrain_model_path))
        if isinstance(self.model, LinearNextLoc):
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

        if isinstance(self.model, LinearNextLoc):
            model_keys = self.model.model.bert.state_dict()
        else:
            raise ValueError('No bert in model!')

        state_dict_load = {}
        unexpect_keys = []
        for k, v in pretrained_model.items():
            if k not in model_keys.keys() or v.shape != model_keys[k].shape\
                    or self._valid_parameter(k):
                unexpect_keys.append(k)
            else:
                state_dict_load[k] = v
        for k, v in model_keys.items():
            if k not in pretrained_model.keys():
                unexpect_keys.append(k)
        self._logger.info("Unexpected keys: {}".format(unexpect_keys))

        if isinstance(self.model, LinearNextLoc):
            self.model.model.bert.load_state_dict(state_dict_load, strict=False)
        else:
            raise ValueError('No bert in model!')
        self._logger.info("Initialize model from {}".format(initial_ckpt))

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_correct = 0  # total top@1 acc for masked elements in epoch
        total_active_elements = 0  # total masked elements in epoch

        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            X, targets, padding_masks, batch_temporal_mat = batch
            # X: (batch_size, padded_length, feat_dim)
            # targets: (batch_size, )
            # padding_masks: (batch_size, padded_length)
            # batch_temporal_mat: (batch_size, padded_length, padded_length)
            X = X.to(self.device)
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            batch_temporal_mat = batch_temporal_mat.to(self.device)

            graph_dict = self.graph_dict

            predictions = self.model(x=X, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                                     graph_dict=graph_dict)  # (batch_size, n_class)

            batch_loss_list = self.criterion(predictions, targets)
            batch_loss = torch.sum(batch_loss_list)
            num_active = len(batch_loss_list)  # batch_size
            mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization

            if self.l2_reg is not None:
                total_loss = mean_loss + self.l2_reg * loss.l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            total_loss = total_loss / self.grad_accmu_steps
            batches_seen += 1

            # with torch.autograd.detect_anomaly():
            total_loss.backward()

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:
                    self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                self.optimizer.zero_grad()

            with torch.no_grad():
                correct = predictions.argmax(dim=-1).eq(targets).sum().item()
                total_correct += correct
                total_active_elements += num_active
                epoch_loss += batch_loss.item()  # add total loss of batch

            post_fix = {
                "mode": "Train",
                "epoch": epoch_idx,
                "iter": i,
                "lr": self.optimizer.param_groups[0]['lr'],
                "loss": mean_loss.item(),
                "acc(%)": total_correct / total_active_elements * 100,
            }
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        total_correct = total_correct / total_active_elements * 100.0
        self._logger.info("Train: expid = {}, Epoch = {}, avg_loss = {}, total_acc = {}%.".format(
            self.exp_id, epoch_idx, epoch_loss, total_correct))
        self._writer.add_scalar('Train loss', epoch_loss, epoch_idx)
        self._writer.add_scalar('Train acc', total_correct, epoch_idx)
        return epoch_loss, total_correct

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        self.model = self.model.eval()
        if mode == 'Test':
            self.evaluator.clear()

        epoch_loss = 0  # total loss of epoch
        total_correct = 0  # total top@1 acc for masked elements in epoch
        total_active_elements = 0  # total masked elements in epoch

        # labels = []
        # preds = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc="{} epoch={}".format(
                    mode, epoch_idx), total=len(eval_dataloader)):
                X, targets, padding_masks, batch_temporal_mat = batch
                # X: (batch_size, padded_length, feat_dim)
                # targets: (batch_size, )
                # padding_masks: (batch_size, padded_length)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X = X.to(self.device)
                targets = targets.to(self.device)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                batch_temporal_mat = batch_temporal_mat.to(self.device)

                predictions = self.model(x=X, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                                         graph_dict=self.graph_dict)  # (batch_size, n_class)

                if mode == 'Test':
                    # preds.append(predictions.cpu().numpy().argmax(axis=-1))
                    # labels.append(targets.cpu().numpy())
                    evaluate_input = {
                        'loc_true': targets.cpu().numpy(),
                        'loc_pred': predictions.cpu().numpy()
                    }
                    self.evaluator.collect(evaluate_input)

                batch_loss_list = self.criterion(predictions, targets)
                batch_loss = torch.sum(batch_loss_list)
                num_active = len(batch_loss_list)  # batch_size
                mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization

                correct = predictions.argmax(dim=-1).eq(targets).sum().item()
                total_correct += correct
                total_active_elements += num_active
                epoch_loss += batch_loss.item()  # add total loss of batch

                post_fix = {
                    "mode": mode,
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "loss": mean_loss.item(),
                    "acc(%)": total_correct / total_active_elements * 100,
                }
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))

            epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
            total_correct = total_correct / total_active_elements * 100.0
            self._logger.info("{}: expid = {}, Epoch = {}, avg_loss = {}, total_acc = {}%.".format(
                mode, self.exp_id, epoch_idx, epoch_loss, total_correct))
            self._writer.add_scalar('{} loss'.format(mode), epoch_loss, epoch_idx)
            self._writer.add_scalar('{} acc'.format(mode), total_correct, epoch_idx)

            if mode == 'Test':
                # preds = np.concatenate(preds, axis=0)
                # labels = np.concatenate(labels, axis=0)
                # np.save(self.cache_dir + '/nextloc_labels.npy', labels)
                # np.save(self.cache_dir + '/nextloc_preds.npy', preds)
                self.evaluator.save_result(self.evaluate_res_dir)
            return epoch_loss, total_correct
