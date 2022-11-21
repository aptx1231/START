import os
import time
import numpy as np
import torch

from libcity.executor.abstract_executor import AbstractExecutor
from libcity.model import loss
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


class BertBaseExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.criterion = torch.nn.NLLLoss(ignore_index=0, reduction='none')
        self.initial_ckpt = self.config.get("initial_ckpt", None)
        self.unload_param = self.config.get("unload_param", [])
        if self.initial_ckpt:
            self.load_model_with_initial_ckpt(self.initial_ckpt)

    def _valid_parameter(self, k):
        for para in self.unload_param:
            if para in k:
                return True
        return False

    def load_model_with_initial_ckpt(self, initial_ckpt):
        assert os.path.exists(initial_ckpt), 'Weights at %s not found' % initial_ckpt
        checkpoint = torch.load(initial_ckpt, map_location='cpu')
        pretrained_model = checkpoint['model'].state_dict()
        model_keys = self.model.state_dict()
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
        self.model.load_state_dict(state_dict_load, strict=False)
        self._logger.info("Initialize model from {}".format(initial_ckpt))

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        start_time = time.time()
        self._valid_epoch(test_dataloader, 0, mode='Test')
        t1 = time.time()
        self._logger.info('Test time {}s.'.format(t1 - start_time))

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = -1
        train_time = []
        eval_time = []
        train_loss = []
        train_acc = []
        eval_loss = []
        eval_acc = []
        lr_list = []

        num_batches = len(train_dataloader)
        self._logger.info("Num_batches: train={}, eval={}".format(num_batches, len(eval_dataloader)))

        for epoch_idx in range(self.epochs):
            start_time = time.time()
            train_avg_loss, train_avg_acc = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss.append(train_avg_loss)
            train_acc.append(train_avg_acc)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            eval_avg_loss, eval_avg_acc = self._valid_epoch(eval_dataloader, epoch_idx, mode='Eval')
            end_time = time.time()
            eval_time.append(end_time - t2)
            eval_loss.append(eval_avg_loss)
            eval_acc.append(eval_avg_acc)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(eval_avg_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(epoch_idx + 1)
                else:
                    self.lr_scheduler.step()

            log_lr = self.optimizer.param_groups[0]['lr']
            lr_list.append(log_lr)
            if (epoch_idx % self.log_every) == 0:
                message = 'Epoch [{}/{}] ({})  train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, (epoch_idx + 1) * num_batches, train_avg_loss,
                           eval_avg_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if eval_avg_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, eval_avg_loss, model_file_name))
                min_val_loss = eval_avg_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

            if (epoch_idx + 1) % self.test_every == 0:
                self.evaluate(test_dataloader)

        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)

        self._draw_png([(train_loss, eval_loss, 'loss'), (train_acc, eval_acc, 'acc'), (lr_list, 'lr')])
        return min_val_loss

    def _draw_png(self, data):
        for data_iter in data:
            plt.figure()
            if len(data_iter) == 3:
                train_list, eval_list, name = data_iter
                x_index = np.arange((len(train_list)))
                plt.plot(x_index, train_list, 'r', label='train_{}'.format(name))
                plt.plot(x_index, eval_list, 'b', label='eval_{}'.format(name))
            else:
                data_list, name = data_iter
                x_index = np.arange((len(data_list)))
                plt.plot(x_index, data_list, 'r', label='{}'.format(name))
            plt.ylabel(name)
            plt.xlabel('epoch')
            plt.title(str(self.exp_id) + ': ' + str(self.model_name))
            plt.legend()
            path = self.png_dir + '/{}_{}.png'.format(self.exp_id, name)
            plt.savefig(path)
            self._logger.info('Save png at {}'.format(path))

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)  # 总batch数

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_correct = 0  # total top@1 acc for masked elements in epoch
        total_active_elements = 0  # total masked elements in epoch

        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            X, targets, target_masks, padding_masks, batch_temporal_mat = batch
            # X/targets/target_masks: (batch_size, padded_length, feat_dim)
            # padding_masks: (batch_size, padded_length)
            # batch_temporal_mat: (batch_size, padded_length, padded_length)
            X = X.to(self.device)
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            batch_temporal_mat = batch_temporal_mat.to(self.device)

            graph_dict = self.graph_dict

            predictions = self.model(x=X, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                                     graph_dict=graph_dict)  # (batch_size, padded_length, n_class)

            targets = targets[..., 0]  # (batch_size, padded_length)
            target_masks = target_masks[..., 0]  # (batch_size, padded_length)

            batch_loss_list = self.criterion(predictions.transpose(1, 2), targets)
            batch_loss = torch.sum(batch_loss_list)
            num_active = target_masks.sum()
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
                mask_label = targets[target_masks]  # (num_active, )
                lm_output = predictions[target_masks].argmax(dim=-1)  # (num_active, )
                assert mask_label.shape == lm_output.shape
                correct = mask_label.eq(lm_output).sum().item()
                total_correct += correct
                total_active_elements += num_active.item()
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

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc="{} epoch={}".format(
                    mode, epoch_idx), total=len(eval_dataloader)):
                X, targets, target_masks, padding_masks, batch_temporal_mat = batch
                # X/targets/target_masks: (batch_size, padded_length, feat_dim)
                # padding_masks: (batch_size, padded_length)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X = X.to(self.device)
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                batch_temporal_mat = batch_temporal_mat.to(self.device)

                predictions = self.model(x=X, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                                         graph_dict=self.graph_dict)  # (batch_size, padded_length, n_class)

                targets = targets[..., 0]  # (batch_size, padded_length)
                target_masks = target_masks[..., 0]  # (batch_size, padded_length)

                if mode == 'Test':
                    evaluate_input = {
                        'loc_true': targets[target_masks].reshape(-1, 1).squeeze(-1).cpu().numpy(),  # (num_active, )
                        'loc_pred': predictions[target_masks].reshape(-1, predictions.shape[-1]).cpu().numpy()  # (num_active, n_class)
                    }
                    self.evaluator.collect(evaluate_input)

                batch_loss_list = self.criterion(predictions.transpose(1, 2), targets)
                batch_loss = torch.sum(batch_loss_list)
                num_active = target_masks.sum()
                mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization

                mask_label = targets[target_masks]  # (num_active, )
                lm_output = predictions[target_masks].argmax(dim=-1)  # (num_active, )
                assert mask_label.shape == lm_output.shape
                correct = mask_label.eq(lm_output).sum().item()
                total_correct += correct
                total_active_elements += num_active.item()
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
                self.evaluator.save_result(self.evaluate_res_dir)
            return epoch_loss, total_correct
